@file:Suppress("unused")

package uy.com.collokia.nlpTest.parser

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import uy.com.collokia.common.utils.elasticSearch.readSoContentFromEs
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.PARSER_TYPE
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizerRaw
import uy.com.collokia.nlp.parser.toNLPContentRDD
import uy.com.collokia.nlpTest.util.LEMMATIZED_INDEX_NAME
import uy.com.collokia.nlpTest.util.RAW_LEMMATIZED_INDEX_NAME
import uy.com.collokia.nlpTest.util.constructTokenizedTestDataset
import java.io.Serializable


class LemmatizerTest : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = LemmatizerTest()
                //test.readLemmatizedContentFromES()
                test.writeLemmatizedContentToES(isRaw = false, isStoreToEs = false)
            }
            println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
        }
    }

    fun writeLemmatizedContentToES(isRaw: Boolean, isStoreToEs: Boolean) {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 4)
        val sparkSession = getLocalSparkSession("Test NLP parser")

        if (isRaw) {
            val testCorpus = constructTokenizedTestDataset(jsc, sparkSession, isRaw = true)
            rawLemmatizerTest(sparkSession, testCorpus, isStoreToEs = isStoreToEs)
        } else {
            val testCorpus = constructTokenizedTestDataset(jsc, sparkSession, isRaw = false)
            lemmatizerTest(sparkSession, testCorpus, isStoreToEs = isStoreToEs)
        }

        closeSpark(jsc)
    }

    fun lemmatizerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>, isStoreToEs: Boolean) {

        val lemmatizer = MateLemmatizer(sparkSession)
        val lemmatized = lemmatizer.transform(testCorpus)

        lemmatized.show(10, false)

        val lemmatizedRDD = toNLPContentRDD(lemmatized, PARSER_TYPE.LEMMATIZER)

        if (isStoreToEs) {
            val lemmatizedDataset = lemmatizedRDD.convertRDDToDF(sparkSession)
            JavaEsSparkSQL.saveToEs(lemmatizedDataset, "$LEMMATIZED_INDEX_NAME/lemmatizedContent")
            //lemmatized?.show(10, false)
        }
    }


    fun rawLemmatizerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>, isStoreToEs: Boolean) {
        val lemmatizer = MateLemmatizerRaw(sparkSession, isRawOutput = true)
        val lemmatizedDataset = lemmatizer.transform(testCorpus)

        lemmatizedDataset.show(10, false)

        if (isStoreToEs) {

            JavaEsSparkSQL.saveToEs(lemmatizedDataset, "$RAW_LEMMATIZED_INDEX_NAME/lemmatizedContent")
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
                .set("es.read.field.as.array.include", "content, content.sentence")

        val jsc = JavaSparkContext(sparkConf)

        jsc.appName()

        val sparkSession = getLocalSparkSession("ES test")
        val documents = readSoContentFromEs(sparkSession, LEMMATIZED_INDEX_NAME)
        documents.show(10, false)

        toNLPContentRDD(documents, PARSER_TYPE.LEMMATIZER).convertRDDToDF(sparkSession).show(10, false)
    }
}
