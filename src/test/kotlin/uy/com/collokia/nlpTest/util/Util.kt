package uy.com.collokia.nlpTest.util

import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import java.io.Serializable
import java.util.*


const val lemmatizedIndexName = "lemmatizer_test"
const val taggedIndexName = "tagger_test"
const val parsedIndexName = "parser_test"

data class TestData(val id: Int, val text: String) : Serializable

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
    val tokenizer = OpenNlpTokenizer(sparkSession, language = LANGUAGE.ENGLISH, isOutputRaw = false)

    val tokenized = tokenizer.transform(input)

    tokenized?.show(10, false)

    return tokenized
}
