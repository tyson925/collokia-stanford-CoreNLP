package uy.com.collokia.nlpTest.util

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizerRaw
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import uy.com.collokia.nlpTest.parser.TestDocument
import java.util.*

const val RAW_LEMMATIZED_INDEX_NAME = "lemmatizer_raw_test"
const val LEMMATIZED_INDEX_NAME = "lemmatizer_test"
const val TAGGED_INDEX_NAME = "tagger_test"
const val PARSED_INDEX_NAME = "parser_test"

fun generateDataSet(jsc: JavaSparkContext): JavaRDD<TestDocument> {
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

fun generatePortugueseDataSet(jsc: JavaSparkContext) : JavaRDD<TestDocument> {
    val test = LinkedList<TestDocument>()

    test.add(TestDocument("1", "O novo compromisso é de três anos, ea apresentação será nesta quarta."))
    test.add(TestDocument("2", "A língua portuguesa, também designada português, é uma língua românica flexiva originada no galego-português falado no Reino da Galiza e no norte de Portugal."))
    test.add(TestDocument("3", "Com a criação do Reino de Portugal em 1139 e a expansão para o sul como parte da Reconquista deu-se a difusão da língua pelas terras conquistadas e mais tarde."))
    test.add(TestDocument("4", "Com as descobertas portuguesas, para o Brasil, África e outras partes do mundo."))

    return jsc.parallelize(test)
}


fun constructTokenizedTestDataset(sparkSession: SparkSession,
                                  testRDD : JavaRDD<TestDocument>,
                                  language: LANGUAGE,
                                  isRaw : Boolean): Dataset<Row> {


    //val testRdd = generateDataSet(jsc)

    val input = sparkSession.createDataFrame(testRDD, TestDocument::class.java).toDF()
    //val sparkSession = SparkSession.builder().master("local").appName("test").orCreate
    val tokenizer = OpenNlpTokenizer(sparkSession, language = language, isOutputRaw = isRaw)

    val tokenized = tokenizer.transform(input)

    tokenized.show(10, false)

    return tokenized
}


fun constructLemmatizedTestDataset(jsc: JavaSparkContext, sparkSession: SparkSession, isRaw: Boolean): Dataset<Row> {
    val testRdd = generateDataSet(jsc)

    val inputData = sparkSession.createDataFrame(testRdd, TestDocument::class.java).toDF()

    val tokenizer = OpenNlpTokenizer(sparkSession, language = LANGUAGE.ENGLISH, isOutputRaw = isRaw, inputColName = "content")
    val lemmatizer = if (isRaw) {
        MateLemmatizerRaw(sparkSession,isRawOutput = false)
    } else {
        MateLemmatizer(sparkSession)
    }

    val pipeline = Pipeline().setStages(arrayOf(tokenizer, lemmatizer))

    return pipeline.fit(inputData).transform(inputData)
}