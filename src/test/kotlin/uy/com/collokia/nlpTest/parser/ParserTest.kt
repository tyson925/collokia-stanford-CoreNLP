package uy.com.collokia.nlpTest.parser

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.mate.parser.ParsedContent
import uy.com.collokia.nlp.parser.mate.parser.ParsedSentence
import uy.com.collokia.nlp.parser.mate.parser.ParsedToken
import uy.com.collokia.nlpTest.util.constructTestDataset
import uy.com.collokia.nlpTest.util.parsedIndexName

class ParserTest() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val test = ParserTest()
            test.writeParsedContentToES()
        }
    }

    fun parserTest(sparkSession: SparkSession, testCorpus: Dataset<Row>) {

        val parser = MateParser(sparkSession)
        val parsedContent = parser.transform(testCorpus)?.toJavaRDD()?.map { row ->
            println(row.schema())
            val parsedSentences = row.getList<WrappedArray<WrappedArray<String>>>(3)
            ParsedContent(parsedSentences.map { sentence ->

                ParsedSentence(JavaConversions.asJavaCollection(sentence).map { token ->
                    ParsedToken(token.apply(0), token.apply(1), token.apply(2), token.apply(3), token.apply(4).toInt())
                })
            })
        }

        val parsedDataset = parsedContent?.convertRDDToDF(sparkSession)
        JavaEsSparkSQL.saveToEs(parsedDataset, "$parsedIndexName/parsedContent")
    }

    fun writeParsedContentToES() {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 2)
        val sparkSession = getLocalSparkSession("Test NLP parser")


        val testCorpus = constructTestDataset(jsc, sparkSession)
        testCorpus?.let {
            parserTest(sparkSession, testCorpus)
        }
        closeSpark(jsc)
    }

}

