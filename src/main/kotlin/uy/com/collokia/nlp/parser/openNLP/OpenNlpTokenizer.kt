package uy.com.collokia.nlp.parser.openNLP

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import java.io.Serializable

data class TokenizedContent(var content : String, var tokenizedContent : List<String>) : Serializable

class OpenNlpTokenizer  : Transformer {

    var tokenizerWrapper : OpenNlpTokenizerWrapper
    var sdetectorWrapper : OpenNlpSentenceDetectorWrapper
    var sparkSession: SparkSession
    var inputColName : String
    var outputColName : String

    constructor(sparkSession: SparkSession,
            sentenceDetectorModelName : String = "./../MLyBigData/NLPUtils/data/opennlp/models/en-sent.bin",
            tokenizerModelName: String = "./../MLyBigData/NLPUtils/data/opennlp/models/en-token.bin"){

        sdetectorWrapper = OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)
        tokenizerWrapper = OpenNlpTokenizerWrapper(tokenizerModelName)
        this.sparkSession = sparkSession
        this.inputColName = "content"
        this.outputColName = TokenizedContent::tokenizedContent.name
    }

    fun setInputColName(inputColName : String) : OpenNlpTokenizer {
        this.inputColName = inputColName
        return this
    }

    override fun uid(): String {
        return "uid1111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return OpenNlpTokenizer(sparkSession)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        return dataset?.let{
            val selectContent = dataset.select("content")
            val beanEncoder = Encoders.bean(TokenizedContent::class.java)
            selectContent.map({row ->
                val text = row.getString(0)
                val tokenizedText = sdetectorWrapper.get().sentDetect(text).flatMap { sentence ->
                    tokenizerWrapper.get().tokenize(sentence).toList()
                }
                TokenizedContent(text,tokenizedText)
            },beanEncoder).toDF()
        } ?: dataset?.toDF()

    }

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes){
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        return SchemaUtils.appendColumn(schema, outputColName,DataTypes.createArrayType(DataTypes.StringType),inputType?.nullable() ?: false)
    }

}