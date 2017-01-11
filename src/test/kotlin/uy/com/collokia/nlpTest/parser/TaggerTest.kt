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
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.mate.tagger.*
import uy.com.collokia.nlp.transformer.candidateNGram.CandidateNGram
import uy.com.collokia.nlp.transformer.candidateNGram.candidateNgramOutputColName
import uy.com.collokia.nlpTest.util.constructTestDataset
import uy.com.collokia.nlpTest.util.taggedIndexName

class TaggerTest() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val test = TaggerTest()
            //test.writeTaggedContentToES()
            val corpus = test.readTaggedContentFromES()
            test.candidateExtractorTest(corpus)
        }
    }

    fun taggerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>) {

        val tagger = MateTagger(sparkSession)
        val taggedContent = tagger.transform(testCorpus)?.toJavaRDD()?.map { row ->
            println(row.schema())
            val parsedSentences = row.getList<WrappedArray<WrappedArray<String>>>(3)
            TaggedContent(parsedSentences.map { sentence ->

                TaggedSentence(JavaConversions.asJavaCollection(sentence).map { token ->
                    TaggedToken(token.apply(0), token.apply(1), token.apply(2))
                })
            })
        }

        //println(lemmatized?.collect()?.joinToString("\n"))
        val taggedDataset = taggedContent?.convertRDDToDF(sparkSession)
        JavaEsSparkSQL.saveToEs(taggedDataset, "$taggedIndexName/taggedContent")
    }

    fun writeTaggedContentToES() {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 2)
        val sparkSession = getLocalSparkSession("Test NLP parser")


        val testCorpus = constructTestDataset(jsc, sparkSession)
        testCorpus?.let {
            taggerTest(sparkSession, testCorpus)
        }
        closeSpark(jsc)
    }

    fun readTaggedContentFromES(): Dataset<Row> {
        val sparkConf = SparkConf().setAppName("appName").setMaster("local[6]")
                //.set("spark.sql.shuffle.partitions", "1")
                .set("es.nodes", "localhost:9200")
                .set("es.nodes.discovery", "true")
                .set("es.nodes.wan.only", "false")
                .set("spark.default.parallelism", "8")
                .set("num-executors", "3")
                .set("executor-cores", "4")
                .set("executor-memory", "4G")
                .set("es.read.field.as.array.include", "taggedContent, taggedContent.taggedSentence")

        val jsc = JavaSparkContext(sparkConf)

        jsc.appName()

        val sparkSession = getLocalSparkSession("ES test")
        val documents = readSoContentFromEs(sparkSession, taggedIndexName)


        val res = documents.toJavaRDD().map { row ->
            //println(row.schema())
            val data = row.getSeq<Row>(0)

            //val data = row.getAs<WrappedArray<WrappedArray<*>>>(0)
            println(data.javaClass.kotlin)
            val res = JavaConversions.asJavaCollection(data)
            TaggedContent(res.map { row1 ->
                val sentence = JavaConversions.asJavaCollection(row1.getSeq<Row>(0))
                TaggedSentence(
                        sentence.map { parsedToken ->
                            val token = parsedToken.getAs<String>("token")
                            val lemma = parsedToken.getAs<String>("lemma")
                            val posTag = parsedToken.getAs<String>("posTag")
                            //arrayOf(token, lemma, posTag)
                            TaggedToken(token, lemma, posTag)
                        })
            })
        }

        //documents.show(10, false)
        return res.convertRDDToDF(sparkSession).toDF()
    }

    fun candidateExtractorTest(dataset: Dataset<Row>) {
        val candidateExtractor = CandidateNGram(LANGUAGE.ENGLISH)
        candidateExtractor.inputCol = taggerOutputColName
        candidateExtractor.outputCol = candidateNgramOutputColName

        val candidates = candidateExtractor.transform(dataset)
        candidates.show(10, false)

    }

}


