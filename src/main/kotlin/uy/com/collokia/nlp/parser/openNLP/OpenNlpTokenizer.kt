@file:Suppress("unused", "ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.parser.openNLP

import com.collokia.resources.OPENNLP_SENTENCE_RESOURCES_PATH_EN
import com.collokia.resources.OPENNLP_SENTENCE_RESOURCES_PATH_ES
import com.collokia.resources.OPENNLP_TOKEN_RESOURCES_PATH_EN
import com.collokia.resources.OPENNLP_TOKEN_RESOURCES_PATH_ES
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
import uy.com.collokia.common.utils.resources.ResourceUtil
import uy.com.collokia.nlp.parser.LANGUAGE
import java.io.Serializable

const val tokenizedContent = "tokenizedContent"

// "opennlp/models/en-sent.bin"
val englishSentenceDetectorModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_SENTENCE_RESOURCES_PATH_EN).absolutePath
}
//= "opennlp/models/es-sent.bin"
val spanishSentenceDetectorModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_SENTENCE_RESOURCES_PATH_ES).absolutePath
}
// "opennlp/models/en-token.bin"
val englishTokenizerModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_TOKEN_RESOURCES_PATH_EN).absolutePath
}
// "opennlp/models/es-token.bin"
val spanishTokenizerModelName: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_TOKEN_RESOURCES_PATH_ES).absolutePath
}


class OpenNlpTokenizer : Transformer, Serializable {

    var tokenizerWrapper: OpenNlpTokenizerWrapper
    var sdetectorWrapper: OpenNlpSentenceDetectorWrapper
    var sparkSession: SparkSession
    var inputColName: String
    var outputColName: String
    val udfName = "tokenizer"
    val language: LANGUAGE
    var isRaw: Boolean

    constructor(sparkSession: SparkSession,
                inputColName: String = "content",
                language: LANGUAGE = LANGUAGE.ENGLISH,
                isOutputRaw: Boolean = true,
                sentenceDetectorModelName: String = englishSentenceDetectorModelName,
                tokenizerModelName: String = englishTokenizerModelName
    ) {

        //SentenceModel()

        this.language = language
        sdetectorWrapper = if (language == LANGUAGE.ENGLISH) OpenNlpSentenceDetectorWrapper(englishSentenceDetectorModelName)
        else if (language == LANGUAGE.SPANISH) OpenNlpSentenceDetectorWrapper(spanishSentenceDetectorModelName)
        else OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)

        tokenizerWrapper = if (language == LANGUAGE.ENGLISH) OpenNlpTokenizerWrapper(englishTokenizerModelName)
        else if (language == LANGUAGE.SPANISH) OpenNlpTokenizerWrapper(spanishTokenizerModelName)
        else OpenNlpTokenizerWrapper(tokenizerModelName)

        this.isRaw = isOutputRaw
        this.sparkSession = sparkSession
        this.inputColName = inputColName
        this.outputColName = tokenizedContent

        if (isOutputRaw) {
            val tokenizer = UDF1 { content: String ->
                val tokenizedText = try {
                    sdetectorWrapper.get().sentDetect(content).flatMap { sentence ->
                        tokenizerWrapper.get().tokenize(sentence).toList()
                    }
                } catch (e: Exception) {
                    println("problem with content: " + content)
                    content.split(Regex("\\W"))
                }
                tokenizedText
            }

            sparkSession.udf().register(udfName, tokenizer, DataTypes.createArrayType(DataTypes.StringType))
        } else {

            val tokenizer = UDF1 { content: String ->
                val tokenizedText = try {
                    sdetectorWrapper.get().sentDetect(content).map { sentence ->
                        tokenizerWrapper.get().tokenize(sentence).toList()
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    println("problem with content: " + content)
                    //content.split(Regex("\\W"))
                    content.split(".").map { sentence ->
                        tokenizerWrapper.get().tokenize(sentence).toList()
                    }
                }
                tokenizedText
            }

            sparkSession.udf().register(udfName, tokenizer, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType)))
        }

    }

    fun setInputColName(inputColName: String): OpenNlpTokenizer {
        this.inputColName = inputColName
        return this
    }

    override fun uid(): String {
        return "uid1111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return OpenNlpTokenizer(sparkSession, inputColName, language, isRaw)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row> {
        return dataset?.let {  dataset.select(dataset.col("*"),
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
        return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.StringType), inputType?.nullable() ?: false)
    }

}