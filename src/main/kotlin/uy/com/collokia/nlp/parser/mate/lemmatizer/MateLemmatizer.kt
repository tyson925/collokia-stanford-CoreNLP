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
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.parser.mate.lemmatizer.LematizerWrapper
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable

const val lemmatizedContentCol = "lemmatizedContent"
const val englishLemmatizerModelName = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model"
const val spanishLemmatizerModelName = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model"

class MateLemmatizer : Transformer, Serializable {


    val lemmatizerWrapper: LematizerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val isRaw: Boolean
    val isRawInput: Boolean
    var isEnglish: Boolean
    val udfName = "lemmatizer"

    constructor(sparkSession: SparkSession,
                isRaw: Boolean,
                isRawInput: Boolean,
                isEnglish: Boolean = true,
                inputColName: String = tokenizedContent,
                outputColName: String = lemmatizedContentCol) {

        this.sparkSession = sparkSession
        this.isRaw = isRaw
        this.isRawInput = isRawInput
        this.isEnglish = isEnglish
        val lemmatizerModel = if (isEnglish) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        this.inputColName = inputColName
        this.outputColName = outputColName
        if (isRawInput) {
            val rawLemmatizer = UDF1({ tokens: WrappedArray<String> ->

                val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                sentenceArray[0] = "<root>"

                (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                val lemmatized = SentenceData09()
                lemmatized.init(sentenceArray)

                if (this.isRaw) {
                    lemmatizerWrapper.get().apply(lemmatized).plemmas.joinToString(" ")
                } else {
                    lemmatizerWrapper.get().apply(lemmatized).plemmas.toList()
                }
            })

            if (isRaw) {
                sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.StringType)
            } else {
                sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.createArrayType(DataTypes.StringType))
            }
        } else {
            val lemmatizer = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
                //val strings = Array(4) { "n = $it" }
sentences
                val results = arrayOfNulls<Array<Array<String>>>(sentences.size())
                (0..sentences.size() - 1).forEach { sentenceNum ->
                    val tokens = sentences.apply(sentenceNum)

                    val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                    sentenceArray[0] = "<root>"

                    (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                    val lemmatized = SentenceData09()
                    lemmatized.init(sentenceArray)
                    this.lemmatizerWrapper.get().apply(lemmatized).plemmas.toList()
                    val lemmas = lemmatized.plemmas

                    val lemmatizedValues = sentenceArray.mapIndexed { tokenIndex, token ->
                        val strings = Array(2) { "n = $it" }
                        strings[0] = token ?: ""
                        strings[1] = lemmas[tokenIndex]
                        strings
                    }.toTypedArray()

                    results[sentenceNum] = lemmatizedValues
                }
                results
            })

            sparkSession.udf().register(udfName, lemmatizer, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType))))
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
        return MateLemmatizer(sparkSession, isRaw, isEnglish, isRawInput, inputColName, outputColName)
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
        return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.StringType), inputType?.nullable() ?: false)
    }


}