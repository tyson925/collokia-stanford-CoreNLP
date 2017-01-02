package uy.com.collokia.nlpTest.parser

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import scala.Tuple2
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.elasticSearch.readSoContentFromEs
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedContent
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedSentence
import uy.com.collokia.nlp.parser.mate.lemmatizer.LemmatizedToken
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import java.io.Serializable
import java.util.*

data class TestData(val id: Int, val text: String) : Serializable

class ParserTest : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            //val jsc = getLocalSparkContext("Test NLP parser", cores = 2)
            //val sparkSession = getLocalSparkSession("Test NLP parser")

            val test = ParserTest()

            /*val testCorpus = test.constructTestDataset(jsc, sparkSession)
            testCorpus?.let {
                test.lemmatizerTest(sparkSession, testCorpus)
            }*/

            test.readLemmtizedContentFromES()

            //closeSpark(jsc)
        }
    }

    fun constructTestDataset(jsc: JavaSparkContext, sparkSession: SparkSession): Dataset<Row>? {
        val test = LinkedList<Tuple2<Int, String>>()
        test.add(Tuple2(1, "Stanford University is located in California. It is a great university."))
        test.add(Tuple2(2, "University of Szeged is,located in Hungary. It is a great university."))
        test.add(Tuple2(3, "Collokia is located in Uruguay."))
        test.add(Tuple2(4, "Collokia is located in Uruguay."))
        test.add(Tuple2(5, "Collokia is located in Uruguay."))
        test.add(Tuple2(6, "University of Szeged is located in Hungary. It is a great university."))
        test.add(Tuple2(7, "University of Szeged is located in Hungary. It is a great university."))
        test.add(Tuple2(8, "Stanford University is located in California. It is a great university."))
        test.add(Tuple2(9, "Stanford University is located in California. It is a great university."))
        test.add(Tuple2(10, "Collokia is,located.In Uruguay."))

        val testRdd = jsc.parallelizePairs(test).map { item ->
            TestData(item._1, item._2)
        }

        val input = sparkSession.createDataFrame(testRdd, TestData::class.java).toDF("id", "content")
        //val sparkSession = SparkSession.builder().master("local").appName("test").orCreate
        val tokenizer = OpenNlpTokenizer(sparkSession, isEnglish = true, isRaw = false)

        val tokenized = tokenizer.transform(input)

        tokenized?.show(10, false)

        return tokenized
    }

    fun lemmatizerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>) {

        val tagger = MateLemmatizer(sparkSession, isRawInput = false, isRaw = true)
        val lemmatized = tagger.transform(testCorpus)?.toJavaRDD()?.map { row ->
            println(row.schema())
            val parsedSentences = row.getList<WrappedArray<WrappedArray<String>>>(3)
            LemmatizedContent(parsedSentences.map { sentence ->

                LemmatizedSentence(JavaConversions.asJavaCollection(sentence).map { token ->
                    LemmatizedToken(token.apply(0), token.apply(1))
                })
            })
        }


        //println(lemmatized?.collect()?.joinToString("\n"))
        val lemmatizedDataset = lemmatized?.convertRDDToDF(sparkSession)
        JavaEsSparkSQL.saveToEs(lemmatizedDataset, "lemmatized_test/lemmatizedContent")
        //lemmatized?.show(10, false)
    }

    fun readLemmtizedContentFromES(){
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
        val documents = readSoContentFromEs(sparkSession, "lemmatized_test")
        documents.show(10, false)

    }
}
