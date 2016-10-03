package uy.com.collokia.nlp.parser.openNLP

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructType
import java.io.Serializable

data class ParsedContent(var content : String,var tokenizedContent : List<String>) : Serializable

class OpenNlpTokenizer  : Transformer {

    var tokenizerWrapper : OpenNlpTokenizerWrapper
    var sdetectorWrapper : OpenNlpSentenceDetectorWrapper
    var sparkSession: SparkSession

    constructor(sparkSession: SparkSession,
            sentenceDetectorModelName : String = "./../MLyBigData/NLPUtils/data/opennlp/models/en-sent.bin",
            tokenizerModelName: String = "./../MLyBigData/NLPUtils/data/opennlp/models/en-token.bin"){

        sdetectorWrapper = OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)
        tokenizerWrapper = OpenNlpTokenizerWrapper(tokenizerModelName)
        this.sparkSession = sparkSession
    }

    override fun uid(): String {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun copy(p0: ParamMap?): Transformer {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        return dataset?.let{
            val selectContent = dataset.select("content")
            val beanEncoder = Encoders.bean(ParsedContent::class.java)
            selectContent.map({row ->
                val text = row.getString(0)
                val tokenizedText = sdetectorWrapper.get().sentDetect(text).flatMap { sentence ->
                    tokenizerWrapper.get().tokenize(sentence).toList()
                }
                ParsedContent(text,tokenizedText)
            },beanEncoder).toDF()
        } ?: dataset?.toDF()

    }

    override fun transformSchema(p0: StructType?): StructType {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

}