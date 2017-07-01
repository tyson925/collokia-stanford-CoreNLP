@file:Suppress("unused")

package uy.com.collokia.nlpTest.parser

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.utils.deleteFileIfExist
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.*
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.PARSER_TYPE
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import uy.com.collokia.nlp.parser.toNLPContentRDD
import uy.com.collokia.nlpTest.util.PARSED_INDEX_NAME
import uy.com.collokia.nlpTest.util.constructTokenizedTestDataset
import uy.com.collokia.nlpTest.util.generateDataSet

class ParserTest {
    companion object {
        const val EDUCAR_PARSED_CORPUS = "./../../../data/dataset/educarCorpus/"
        const val EDUCAR_CORPUS = "./../../../../collokia-data-es-indexer/data/educar/textos.json"
        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = ParserTest()
                test.writeParsedContentToES()
                //test.parseEducarCorpus()
            }
            println("Execution time is ${time.second} seconds.")
        }
    }

    fun writeParsedContentToES() {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 1)
        val sparkSession = getLocalSparkSession("Test NLP parser")


        val testCorpus = constructTokenizedTestDataset(sparkSession, generateDataSet(jsc), isRaw = false, language = LANGUAGE.ENGLISH)

        parserTest(sparkSession, testCorpus)

        closeSpark(jsc)
    }

    fun parserTest(sparkSession: SparkSession, testCorpus: Dataset<Row>) {

        val parser = MateParser(sparkSession, language = LANGUAGE.ENGLISH,inputColName = TestDocument::content.name)
        val parsedContent = parser.transform(testCorpus)

        val parsedContentRDD = toNLPContentRDD(parsedContent,PARSER_TYPE.PARSER)

        val parsedDataset = parsedContentRDD.convertRDDToDF(sparkSession)
        JavaEsSparkSQL.saveToEs(parsedDataset, "$PARSED_INDEX_NAME/parsedContent")
    }

    fun parseEducarCorpus() {
        val jsc = getLocalSparkContext("educar", cores = 4)
        val sparkSession = getLocalSparkSession("educar")

        val corpus = jsc.textFile(EDUCAR_CORPUS).map { line ->
            OBJECT_MAPPER.readValue(line, SimpleDocument::class.java)
        }.convertRDDToDF(sparkSession).toDF()
        println(corpus.count())
        corpus.show(10, false)

        //val document = La jornada se realiza con motivo del 30° aniversario del Área Educación de FLACSO. Por este motivo invitan a aquellos investigadores en formación que quieran dar a conocer sus trabajos, a enviar sus resúmenes.

        val tokenizer = OpenNlpTokenizer(sparkSession, SimpleDocument::content.name, LANGUAGE.SPANISH, isOutputRaw = false)
        val lemmatizer = MateLemmatizer(sparkSession, inputColName = tokenizer.outputColName, language = LANGUAGE.SPANISH)

        val tokenized = tokenizer.transform(corpus)
//tokenized?.show(4000)

        val lemmatized = lemmatizer.transform(tokenized)
        lemmatized.show(false)
        //val postTagedCorpus = lemmatizeContent(sparkSession, corpus, SimpleDocument::content.name, LANGUAGE.SPANISH)
        //postTagedCorpus?.show(false)
        deleteFileIfExist(EDUCAR_PARSED_CORPUS)
        lemmatized.write().save(EDUCAR_PARSED_CORPUS)
    }

}

