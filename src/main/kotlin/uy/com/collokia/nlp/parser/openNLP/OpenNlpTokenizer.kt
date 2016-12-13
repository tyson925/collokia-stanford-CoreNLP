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

const val tokenizedContent = "tokenizedContent"

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
        this.outputColName = tokenizedContent

        val tokenizer = UDF1{ content : String ->
            val tokenizedText = sdetectorWrapper.get().sentDetect(content).flatMap { sentence ->
                tokenizerWrapper.get().tokenize(sentence).toList()
            }
            tokenizedText
        }

        sparkSession.udf().register("tokenizer",tokenizer,DataTypes.createArrayType(DataTypes.StringType))

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
        //dataset?.show(10,false)

        return dataset?.select(dataset.col("*"),
                functions.callUDF("tokenizer",JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))

    }

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        //val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes){
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        return SchemaUtils.appendColumn(schema, outputColName,DataTypes.createArrayType(DataTypes.StringType),inputType?.nullable() ?: false)
    }

}