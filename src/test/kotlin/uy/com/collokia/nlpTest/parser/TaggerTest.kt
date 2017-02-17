package uy.com.collokia.nlpTest.parser

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark.sql.api.java.JavaEsSparkSQL
import scala.collection.JavaConversions
import uy.com.collokia.common.utils.elasticSearch.readSoContentFromEs
import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.PARSER_TYPE
import uy.com.collokia.nlp.parser.mate.tagger.*
import uy.com.collokia.nlp.parser.toNLPContentRDD
import uy.com.collokia.nlp.transformer.candidateNGram.CANDIDATE_NGRAM_OUTPUT_COL_NAME
import uy.com.collokia.nlp.transformer.candidateNGram.CandidateNGram
import uy.com.collokia.nlpTest.util.TAGGED_INDEX_NAME
import uy.com.collokia.nlpTest.util.constructTokenizedTestDataset
import uy.com.collokia.nlpTest.util.generateDataSet

class TaggerTest() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val test = TaggerTest()
                test.writeTaggedContentToES()
                //val corpus = test.readTaggedContentFromES()
                //test.candidateExtractorTest(corpus)
            }
            println("Execution time is ${time.second}")
        }
    }

    fun taggerTest(sparkSession: SparkSession, testCorpus: Dataset<Row>, isStoreToEs : Boolean) {

        val tagger = MateTagger(sparkSession)
        val taggedContent = tagger.transform(testCorpus)

        taggedContent.show(10, false)

        if (isStoreToEs) {
            val taggedRDD = toNLPContentRDD(taggedContent, PARSER_TYPE.POSTAGGER)
            val taggedDataset = taggedRDD.convertRDDToDF(sparkSession)
            taggedDataset.show(10,false)
            JavaEsSparkSQL.saveToEs(taggedDataset, "$TAGGED_INDEX_NAME/taggedContent")
        }
    }

    fun writeTaggedContentToES() {
        val jsc = getLocalSparkContext("Test NLP parser", cores = 2)
        val sparkSession = getLocalSparkSession("Test NLP parser")


        val testCorpus = constructTokenizedTestDataset(sparkSession, generateDataSet(jsc), isRaw = false, language = LANGUAGE.ENGLISH)
        taggerTest(sparkSession, testCorpus, isStoreToEs = true)

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
        val documents = readSoContentFromEs(sparkSession, TAGGED_INDEX_NAME)


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
        candidateExtractor.inputCol = TAGGER_OUTPUT_COL_NAME
        candidateExtractor.outputCol = CANDIDATE_NGRAM_OUTPUT_COL_NAME

        val candidates = candidateExtractor.transform(dataset)
        candidates.show(10, false)

    }

}


