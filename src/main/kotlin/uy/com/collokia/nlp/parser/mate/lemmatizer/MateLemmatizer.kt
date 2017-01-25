@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.lemmatizer

import com.collokia.resources.MATE_LEMMATIZER_RESOURCES_PATH_EN
import com.collokia.resources.MATE_LEMMATIZER_RESOURCES_PATH_ES
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
import uy.com.collokia.common.utils.resources.ResourceUtil
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.NLPToken
import uy.com.collokia.nlp.parser.nlpTokenType
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable
import java.util.*

const val lemmatizedContentCol = "lemmatizedContent"
//"mate/models/english/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model"
val englishLemmatizerModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(MATE_LEMMATIZER_RESOURCES_PATH_EN).absolutePath
}
//"mate/models/spanish/CoNLL2009-ST-Spanish-ALL.anna-3.3.lemmatizer.model"
val spanishLemmatizerModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(MATE_LEMMATIZER_RESOURCES_PATH_ES).absolutePath
}

class MateLemmatizer : Transformer, Serializable {

    val lemmatizerWrapper: LematizerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    var language: LANGUAGE
    val udfName = "lemmatizer"

    constructor(sparkSession: SparkSession,
                language: LANGUAGE = LANGUAGE.ENGLISH,
                inputColName: String = tokenizedContent,
                outputColName: String = lemmatizedContentCol) {

        this.sparkSession = sparkSession
        this.language = language
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        this.inputColName = inputColName
        this.outputColName = outputColName


        val lemmatizerUDF = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
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
            results
        })
        //val outputType = outputType(false)

        sparkSession.udf().register(udfName, lemmatizerUDF, DataTypes.createArrayType(DataTypes.createArrayType(nlpTokenType())))

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
        return MateLemmatizer(sparkSession, language, inputColName, outputColName)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row> {

        //val outputDataType = transformSchema(dataset?.schema()).apply(outputColName).metadata()

        return dataset?.let {
            dataset.select(dataset.col("*"),
                    functions.callUDF(udfName, JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
        } ?: sparkSession.emptyDataFrame()
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

    /*private fun outputType(schema: StructType?): StructField {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val nullable = inputType?.nullable() ?: false
        return outputType(nullable)
    }*/


}
