@file:Suppress("unused", "ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.parser.openNLP

import com.collokia.resources.*
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
import uy.com.collokia.nlp.parser.getSentenceSplitterModelName
import uy.com.collokia.nlp.parser.getTokenizerModelName
import java.io.Serializable

const val TOKENIZED_CONTENT_COL_NAME = "tokenizedContenT"

// "opennlp/models/en-sent.bin"
val EN_SENTENCE_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_SENTENCE_RESOURCES_PATH_EN).absolutePath
}
//= "opennlp/models/es-sent.bin"
val ES_SENTENCE_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_SENTENCE_RESOURCES_PATH_ES).absolutePath
}
//= "opennlp/models/pt-sent.bin"
val PT_SENTENCE_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_SENTENCE_RESOURCES_PATH_PT).absolutePath
}

// "opennlp/models/en-token.bin"
val EN_TOKEN_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_TOKEN_RESOURCES_PATH_EN).absolutePath
}
// "opennlp/models/es-token.bin"
val ES_TOKEN_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_TOKEN_RESOURCES_PATH_ES).absolutePath
}

val PT_TOKEN_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(OPENNLP_TOKEN_RESOURCES_PATH_PT).absolutePath
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
                sentenceDetectorModelName: String = "",
                tokenizerModelName: String = ""
    ) {

        this.language = language
        sdetectorWrapper = if (sentenceDetectorModelName.isNotEmpty()) {
            OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)
        } else {
            OpenNlpSentenceDetectorWrapper(getSentenceSplitterModelName(language))
        }

        tokenizerWrapper = if (tokenizerModelName.isNotEmpty()) {
            OpenNlpTokenizerWrapper(tokenizerModelName)
        } else {
            OpenNlpTokenizerWrapper(getTokenizerModelName(language))
        }

        this.isRaw = isOutputRaw
        this.sparkSession = sparkSession
        this.inputColName = inputColName
        this.outputColName = TOKENIZED_CONTENT_COL_NAME

        if (isOutputRaw) {
            val tokenizer = UDF1 { content: String ->
                val tokenizedText = try {
                    sdetectorWrapper.get().sentDetect(content).flatMap { sentence ->
                        tokenizerWrapper.get().tokenize(sentence).toList()
                    }
                } catch (e: Exception) {
                    //println(e.printStackTrace())
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

    fun setOutputColName(outputColName : String) : OpenNlpTokenizer {
        this.outputColName = outputColName
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