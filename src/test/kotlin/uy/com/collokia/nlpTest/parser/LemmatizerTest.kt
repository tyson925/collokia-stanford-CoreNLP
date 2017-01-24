package uy.com.collokia.nlpTest.parser

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.elasticSearch.readSoContentFromEs
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedContent
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedSentence
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedToken
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlpTest.util.constructTokenizedTestDataset
import uy.com.collokia.nlpTest.util.lemmatizedIndexName
import java.io.Serializable


class LemmatizerTest : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = LemmatizerTest()
                //test.readLemmatizedContentFromES()
                test.writeLemmatizedContentToES()
            }
            println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
        }
    }

    fun writeLemmatizedContentToES() {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 4)
        val sparkSession = getLocalSparkSession("Test NLP parser")


        val testCorpus = constructTokenizedTestDataset(jsc, sparkSession)
        testCorpus?.let {
            lemmatizerTest(sparkSession, testCorpus, isStoreToEs = false)
        }
        closeSpark(jsc)
    }

    fun lemmatizerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>, isStoreToEs: Boolean) {

        val tagger = MateLemmatizer(sparkSession, isRawInput = false, isRawOutput = false)
        val lemmatized = tagger.transform(testCorpus)!!

        val lemmatizedRDD = lemmatized.toJavaRDD().map { row ->
            println(row.schema())
            val parsedSentences = row.getList<WrappedArray<WrappedArray<String>>>(3)
            LemmatizedContent(parsedSentences.map { sentence ->

                LemmatizedSentence(JavaConversions.asJavaCollection(sentence).map { token ->
                    LemmatizedToken(token.apply(0), token.apply(1))
                })
            })
        }

        if (isStoreToEs) {
            //println(lemmatized?.collect()?.joinToString("\n"))
            val lemmatizedDataset = lemmatizedRDD.convertRDDToDF(sparkSession)
            JavaEsSparkSQL.saveToEs(lemmatizedDataset, "$lemmatizedIndexName/lemmatizedContent")
            //lemmatized?.show(10, false)
        }
    }


    fun readLemmatizedContentFromES() {
        val sparkConf = SparkConf().setAppName("appName").setMaster("local[6]")
                //.set("spark.sql.shuffle.partitions", "1")
                .set("es.nodes", "localhost:9200")
                .set("es.nodes.discovery", "true")
                .set("es.nodes.wan.only", "false")
                .set("spark.default.parallelism", "8")
                .set("num-executors", "3")
                .set("executor-cores", "4")
                .set("executor-memory", "4G")
                .set("es.read.field.as.array.include", "lemmatizedContent, lemmatizedContent.lemmatizedSentence")

        val jsc = JavaSparkContext(sparkConf)

        jsc.appName()

        val sparkSession = getLocalSparkSession("ES test")
        val documents = readSoContentFromEs(sparkSession, lemmatizedIndexName)
        documents.show(10, false)

    }
}
