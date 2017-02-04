@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.data.SentenceData09
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.nospark.NoSparkTransformer
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.NLPToken
import uy.com.collokia.nlp.parser.nlpTokenType
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable
import java.util.*

class NoSparkMateLemmatizer(val sparkSession: SparkSession, var language: LANGUAGE = LANGUAGE.ENGLISH, var inputColName: String = tokenizedContent, var outputColName: String = lemmatizedContentCol) : NoSparkTransformer(), Serializable {
    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkMateLemmatizer>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkMateLemmatizer")

    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        val sentences = mapIn[inputColName] as WrappedArray<WrappedArray<String>>
        val lemmatizer = lemmatizerWrapper.get()
        val results = ArrayList<Array<Map<String, String>>>(sentences.size())

        (0..sentences.size() - 1).forEach { sentenceNum ->

            val tokens = sentences.apply(sentenceNum)

            val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

            val lemmatized = SentenceData09()
            lemmatized.init(sentenceArray)
            lemmatizer.apply(lemmatized)
            val lemmas = lemmatized.plemmas

            val contentIndex = results.map { sentence -> sentence.size }.sum()

            val lemmatizedTokens = sentenceArray.mapIndexed { tokenIndex, token ->

                val values = mapOf(
                        NLPToken::index.name to tokenIndex.toString(),
                        NLPToken::token.name to (token ?: ""),
                        NLPToken::lemma.name to lemmas[tokenIndex],
                        NLPToken::indexInContent.name to (contentIndex + tokenIndex).toString()
                )
                values
            }.toTypedArray()

            results.add(sentenceNum, lemmatizedTokens)
        }
        return mapIn + (outputColName to results)

    }

    val lemmatizerWrapper: LematizerWrapper
    val udfName = "lemmatizer"


    fun setInputColName(inputColName: String): NoSparkMateLemmatizer {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): NoSparkMateLemmatizer {
        this.outputColName = outputColName
        return this
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
        //return SchemaUtils.appendColumn(schema, DataTypes.createStructField(outputColName, DataTypes.StringType, nullable))
    }

    private fun outputType(nullable: Boolean): StructField {
        return DataTypes.createStructField(outputColName, nlpTokenType(), nullable)
    }

    init {
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)
    }
}
