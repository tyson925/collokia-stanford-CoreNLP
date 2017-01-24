package uy.com.collokia.nlpTest.parser

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.utils.deleteFileIfExist
import uy.com.collokia.common.utils.formatterToTimePrint
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.*
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.mate.parser.ParsedContent
import uy.com.collokia.nlp.parser.mate.parser.ParsedSentence
import uy.com.collokia.nlp.parser.mate.parser.ParsedToken
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import uy.com.collokia.nlpTest.util.constructTokenizedTestDataset
import uy.com.collokia.nlpTest.util.parsedIndexName

class ParserTest() {
    companion object {
        const val EDUCAR_PARSED_CORPUS = "./../../../data/dataset/educarCorpus/"
        const val EDUCAR_CORPUS = "./../../../../collokia-data-es-indexer/data/educar/textos.json"
        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = ParserTest()
                //test.writeParsedContentToES()
                test.parseEducarCorpus()
            }
            println("Execution time is ${formatterToTimePrint.format(time.second / 1000.toLong())} seconds.")
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


        val testCorpus = constructTokenizedTestDataset(jsc, sparkSession)
        testCorpus?.let {
            parserTest(sparkSession, testCorpus)
        }
        closeSpark(jsc)
    }

    fun parseEducarCorpus() {
        val jsc = getLocalSparkContext("educar",cores = 4)
        val sparkSession = getLocalSparkSession("educar")

        val corpus = jsc.textFile(EDUCAR_CORPUS).map { line ->
            OBJECT_MAPPER.readValue(line, SimpleDocument::class.java)
        }.convertRDDToDF(sparkSession).toDF()
        println(corpus.count())
        corpus.show(10,false)

        //val document = La jornada se realiza con motivo del 30° aniversario del Área Educación de FLACSO. Por este motivo invitan a aquellos investigadores en formación que quieran dar a conocer sus trabajos, a enviar sus resúmenes.

        val tokenizer = OpenNlpTokenizer(sparkSession,SimpleDocument::content.name, LANGUAGE.SPANISH,isOutputRaw = false)
        val lemmatizer = MateLemmatizer(sparkSession,isRawOutput = false,isRawInput = false,inputColName = tokenizer.outputColName,language = LANGUAGE.SPANISH)

        val tokenized = tokenizer.transform(corpus)
//tokenized?.show(4000)

        val lemmatized = lemmatizer.transform(tokenized)!!
        lemmatized?.show(false)
        //val postTagedCorpus = lemmatizeContent(sparkSession, corpus, SimpleDocument::content.name, LANGUAGE.SPANISH)
        //postTagedCorpus?.show(false)
        deleteFileIfExist(EDUCAR_PARSED_CORPUS)
        lemmatized.write().save(EDUCAR_PARSED_CORPUS)
    }

}

