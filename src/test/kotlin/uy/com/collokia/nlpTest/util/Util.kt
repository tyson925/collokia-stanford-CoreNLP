package uy.com.collokia.nlpTest.util

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import uy.com.collokia.nlpTest.parser.TestDocument
import java.util.*


const val lemmatizedIndexName = "lemmatizer_test"
const val taggedIndexName = "tagger_test"
const val parsedIndexName = "parser_test"

private fun generateDataSet(jsc : JavaSparkContext) : JavaRDD<TestDocument> {
    val test = LinkedList<TestDocument>()
    test.add(TestDocument("1", "Stanford University is located in California. It is a great university."))
    test.add(TestDocument("2", "University of Szeged is,located in Hungary. It is a great university."))
    test.add(TestDocument("3", "Collokia is located in Uruguay."))
    test.add(TestDocument("4", "Collokia is located in Uruguay."))
    test.add(TestDocument("5", "Collokia is located in Uruguay."))
    test.add(TestDocument("6", "University of Szeged is located in Hungary. It is a great university."))
    test.add(TestDocument("7", "University of Szeged is located in Hungary. It is a great university."))
    test.add(TestDocument("8", "Stanford University is located in California. It is a great university."))
    test.add(TestDocument("9", "Stanford University is located in California. It is a great university."))
    test.add(TestDocument("10", "Collokia is,located.In Uruguay."))

    return jsc.parallelize(test)
}


fun constructTokenizedTestDataset(jsc: JavaSparkContext, sparkSession: SparkSession): Dataset<Row> {


    val testRdd = generateDataSet(jsc)

    val input = sparkSession.createDataFrame(testRdd, TestDocument::class.java).toDF("id", "content")
    //val sparkSession = SparkSession.builder().master("local").appName("test").orCreate
    val tokenizer = OpenNlpTokenizer(sparkSession, language = LANGUAGE.ENGLISH, isOutputRaw = false)

    val tokenized = tokenizer.transform(input)

    tokenized.show(10, false)

    return tokenized
}


fun constructLemmatizedTestDataset(jsc: JavaSparkContext, sparkSession: SparkSession, isRaw : Boolean): Dataset<Row> {
    val testRdd = generateDataSet(jsc)

    val inputData = sparkSession.createDataFrame(testRdd, TestDocument::class.java).toDF()

    val tokenizer = OpenNlpTokenizer(sparkSession, language = LANGUAGE.ENGLISH, isOutputRaw = isRaw, inputColName = "content")
    val lemmatizer = MateLemmatizer(sparkSession,isRawInput = isRaw,isRawOutput = false)

    val pipeline = Pipeline().setStages(arrayOf(tokenizer,lemmatizer))

    return pipeline.fit(inputData).transform(inputData)
}