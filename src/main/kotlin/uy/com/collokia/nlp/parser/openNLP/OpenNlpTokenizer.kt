@file:Suppress("unused", "ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.parser.openNLP

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
import java.io.Serializable

const val tokenizedContent = "tokenizedContent"
private const val englishSentenceDetectorModelName = "./../MLyBigData/NLPUtils/data/opennlp/models/en-sent.bin"
private const val spanishSentenceDetectorModelName = "./../MLyBigData/NLPUtils/data/opennlp/models/es-sent.bin"
private const val englishTokenizerModelName = "./../MLyBigData/NLPUtils/data/opennlp/models/en-token.bin"
private const val spanishTokenizerModelName = "./../MLyBigData/NLPUtils/data/opennlp/models/es-token.bin"

class OpenNlpTokenizer : Transformer, Serializable {

    var tokenizerWrapper: OpenNlpTokenizerWrapper
    var sdetectorWrapper: OpenNlpSentenceDetectorWrapper
    var sparkSession: SparkSession
    var inputColName: String
    var outputColName: String
    val udfName = "tokenizer"
    val isEnglish: Boolean
    var isRaw: Boolean

    constructor(sparkSession: SparkSession, inputColName: String = "content", isEnglish: Boolean = true, isRaw: Boolean = true
            //sentenceDetectorModelName : String = englishSentenceDetectorModelName,
            //tokenizerModelName: String = englishTokenizerModelName
    ) {
        this.isEnglish = isEnglish
        sdetectorWrapper = if (isEnglish) OpenNlpSentenceDetectorWrapper(englishSentenceDetectorModelName) else OpenNlpSentenceDetectorWrapper(spanishSentenceDetectorModelName)
        tokenizerWrapper = if (isEnglish) OpenNlpTokenizerWrapper(englishTokenizerModelName) else OpenNlpTokenizerWrapper(spanishTokenizerModelName)
        this.isRaw = isRaw
        this.sparkSession = sparkSession
        this.inputColName = inputColName
        this.outputColName = tokenizedContent

        if (isRaw) {
            val tokenizer = UDF1 { content: String ->
                val tokenizedText = sdetectorWrapper.get().sentDetect(content).flatMap { sentence ->
                    tokenizerWrapper.get().tokenize(sentence).toList()
                }
                tokenizedText
            }

            sparkSession.udf().register(udfName, tokenizer, DataTypes.createArrayType(DataTypes.StringType))
        } else {
            val tokenizer = UDF1 { content: String ->
                val tokenizedText = sdetectorWrapper.get().sentDetect(content).map { sentence ->
                    tokenizerWrapper.get().tokenize(sentence).toList()
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
        return OpenNlpTokenizer(sparkSession, inputColName, isEnglish, isRaw)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {
        //dataset?.show(10,false)

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