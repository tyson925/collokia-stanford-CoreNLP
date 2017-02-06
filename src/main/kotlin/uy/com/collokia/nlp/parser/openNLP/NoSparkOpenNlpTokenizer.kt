@file:Suppress("unused", "ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.parser.openNLP

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.utils.nospark.NoSparkTransformer
import uy.com.collokia.nlp.parser.LANGUAGE
import java.io.Serializable


class NoSparkOpenNlpTokenizer(var sparkSession: SparkSession,
                              var inputColName: String = SimpleDocument::content.name,
                              val language: LANGUAGE = LANGUAGE.ENGLISH,
                              val isOutputRaw: Boolean = true,
                              val outputColName:String = "tokenizedContent",
                              sentenceDetectorModelName: String = englishSentenceDetectorModelName,
                              tokenizerModelName: String = englishTokenizerModelName
) : NoSparkTransformer(), Serializable {

    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        val content = mapIn[inputColName] as String
        return mapIn + (
                outputColName to
                        (if (isOutputRaw) {
                            transformSingleRaw(content)
                        } else {
                            transformSingle(content).map{ it.toTypedArray()}
                        }).toTypedArray()
                )
    }


    fun transformSingleRaw(content: String): List<String> {


        val tokenizedText = try {
            sdetectorWrapper.get().sentDetect(content).flatMap { sentence ->
                tokenizerWrapper.get().tokenize(sentence).toList()
            }
        } catch (e: Exception) {
            println("problem with content: " + content)
            content.split(Regex("\\W"))
        }
        return tokenizedText

    }

    fun transformSingle(content: String): List<List<String>> {

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
        return tokenizedText

    }


    var tokenizerWrapper: OpenNlpTokenizerWrapper = if (language == LANGUAGE.ENGLISH) OpenNlpTokenizerWrapper(englishTokenizerModelName)
    else if (language == LANGUAGE.SPANISH) OpenNlpTokenizerWrapper(spanishTokenizerModelName)
    else OpenNlpTokenizerWrapper(tokenizerModelName)
    var sdetectorWrapper: OpenNlpSentenceDetectorWrapper = if (language == LANGUAGE.ENGLISH) OpenNlpSentenceDetectorWrapper(englishSentenceDetectorModelName)
    else if (language == LANGUAGE.SPANISH) OpenNlpSentenceDetectorWrapper(spanishSentenceDetectorModelName)
    else OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)
    val udfName = "tokenizer"
    var isRaw = isOutputRaw

    fun setInputColName(inputColName: String): NoSparkOpenNlpTokenizer {
        this.inputColName = inputColName
        return this
    }

    override fun uid(): String {
        return "uid1111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return OpenNlpTokenizer(sparkSession, inputColName, language, isRaw)
    }

    /* override fun transform(dataset: Dataset<*>?): Dataset<Row> {
         return dataset?.let {  dataset.select(dataset.col("*"),
                 functions.callUDF(udfName, JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
         } ?: sparkSession.emptyDataFrame()
     }
     */

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        //val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes) {
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        if (isRaw) {
            return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.StringType), inputType?.nullable() ?: false)
        }else{
            return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType)), inputType?.nullable() ?: false)
        }
    }

}