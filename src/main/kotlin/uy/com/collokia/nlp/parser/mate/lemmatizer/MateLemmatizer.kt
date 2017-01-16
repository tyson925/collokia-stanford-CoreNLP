@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.data.SentenceData09
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.functions
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable
import java.util.*

const val lemmatizedContentCol = "lemmatizedContent"
const val englishLemmatizerModelName = "mate/models/english/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model"
const val spanishLemmatizerModelName = "mate/models/spanish/CoNLL2009-ST-Spanish-ALL.anna-3.3.lemmatizer.model"

data class LemmatizedToken(var token: String, var lemma: String) : Serializable

data class LemmatizedSentence(var lemmatizedSentence: List<LemmatizedToken>) : Serializable

data class LemmatizedContent(var lemmatizedContent : List<LemmatizedSentence>) : Serializable

class MateLemmatizer : Transformer, Serializable {


    val lemmatizerWrapper: LematizerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val isRawOutput: Boolean
    val isRawInput: Boolean
    var language: LANGUAGE
    val udfName = "lemmatizer"

    constructor(sparkSession: SparkSession,
                isRawOutput: Boolean,
                isRawInput: Boolean,
                language: LANGUAGE = LANGUAGE.ENGLISH,
                inputColName: String = tokenizedContent,
                outputColName: String = lemmatizedContentCol) {

        this.sparkSession = sparkSession
        this.isRawOutput = isRawOutput
        this.isRawInput = isRawInput
        this.language = language
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        this.inputColName = inputColName
        this.outputColName = outputColName

        if (isRawInput) {
            val rawLemmatizer = UDF1({ tokens: WrappedArray<String> ->
                val lemmatizer = lemmatizerWrapper.get()
                val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                sentenceArray[0] = "<root>"

                (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                val lemmatized = SentenceData09()
                lemmatized.init(sentenceArray)

                if (this.isRawOutput) {
                    lemmatizer.apply(lemmatized).plemmas.joinToString(" ")
                } else {
                    lemmatizer.apply(lemmatized).plemmas.toList()
                }
            })

            if (isRawOutput) {
                sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.StringType)
            } else {
                sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.createArrayType(DataTypes.StringType))
            }
        } else {
            val lemmatizerUDF = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
                val lemmatizer = lemmatizerWrapper.get()
                //val results = arrayOfNulls<Array<Row>>(sentences.size())
                val results = ArrayList<Array<Array<String>>>(sentences.size())

                (0..sentences.size() - 1).forEach { sentenceNum ->

                    val tokens = sentences.apply(sentenceNum)

                    val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                    sentenceArray[0] = "<root>"

                    (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                    val lemmatized = SentenceData09()
                    lemmatized.init(sentenceArray)
                    lemmatizer.apply(lemmatized)
                    val lemmas = lemmatized.plemmas

                    val lemmatizedTokens = sentenceArray.mapIndexed { tokenIndex, token ->
                        arrayOf(token ?: "", lemmas[tokenIndex])
                        //RowFactory.create(token ?: "", lemmas[tokenIndex])
                        //LemmatizedToken(token ?: "", lemmas[tokenIndex])
                    }.toTypedArray()

                    results.add(sentenceNum,lemmatizedTokens)
                }
                results
            })
            val outputType = outputType(false)

            sparkSession.udf().register(udfName, lemmatizerUDF, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType))))
            //sparkSession.udf().register(udfName, lemmatizerUDF, outputType.dataType())
        }
    }


    fun setInputColName(inputColName: String): MateLemmatizer {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): MateLemmatizer {
        this.outputColName = outputColName
        return this
    }

    override fun uid(): String {
        return "lemmatizer111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateLemmatizer(sparkSession, isRawOutput, isRawInput, language, inputColName, outputColName)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        val outputDataType = transformSchema(dataset?.schema()).apply(outputColName).metadata()

        return dataset?.select(dataset.col("*"),
                functions.callUDF(udfName, JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
    }

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        //val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes) {
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        val nullable = inputType?.nullable() ?: false
        val output = outputType(nullable)
        return SchemaUtils.appendColumn(schema, output)
    }

    private fun outputType(nullable: Boolean): StructField {
        val token = DataTypes.createStructField("token", DataTypes.StringType, nullable)
        val lemma = DataTypes.createStructField("lemma", DataTypes.StringType, nullable)
        val output = DataTypes.createStructField(outputColName, DataTypes.createArrayType(DataTypes.createStructType(listOf(token, lemma))), nullable)
        return output
    }

    private fun outputType(schema: StructType?): StructField {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val nullable = inputType?.nullable() ?: false
        return outputType(nullable)
    }


}