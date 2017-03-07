package uy.com.collokia.nlpTest.transformer

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.mate.lemmatizer.LEMMATIZED_CONTENT_COL_NAME
import uy.com.collokia.nlp.transformer.ngram.NGramOnSentenceData
import uy.com.collokia.nlpTest.util.constructLemmatizedTestDataset
import java.io.Serializable


class NGramTransformerTest : Serializable {

    companion object {

        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = NGramTransformerTest()
                val jsc = getLocalSparkContext("text")
                val sparkSession = getLocalSparkSession("text")
                val testData = constructLemmatizedTestDataset(jsc, sparkSession, isRaw = false)

                test.nGramTest(testData)
            }
            println("Execution time is ${time.second} seconds.")
        }
    }

    fun nGramTest(dataset: Dataset<Row>) {
        val ngram = NGramOnSentenceData().setInputCol(LEMMATIZED_CONTENT_COL_NAME)
        dataset.show(10, false)
        val ngramsContent = ngram.transform(dataset)
        ngramsContent.show(10, false)
    }

}

