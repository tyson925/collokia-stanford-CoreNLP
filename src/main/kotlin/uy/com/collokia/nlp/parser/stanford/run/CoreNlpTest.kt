package uy.com.collokia.nlp.parser.stanford.run


import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.SparkSession
import scala.Serializable
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.nlp.parser.stanford.coreNLP.CoreNLP

class TestData(val id: Int, val category : String, val categoryIndex : Double, val content: String) : Serializable


@Suppress("UNCHECKED_CAST") class CoreNlpTest() {

    fun testCoreNlp(jsc: JavaSparkContext, sparkSession: SparkSession) {

        val test = listOf(TestData(1,"bigData",1.0,"<xml>Stanford University is located in California. It is a great university.</xml>"),
                TestData(1,"bigData",1.0,"<xml>University of Szeged is located in Hungary. It is a great university.</xml>"),
                TestData(1,"bigData",1.0,"<xml>Collokia is located in Uruguay.</xml>"),
                TestData(1,"bigData",1.0,"<xml>Collokia is located in Uruguay.</xml>"),
                TestData(1,"bigData",1.0,"<xml>University of Szeged is located in Hungary. It is a great university.</xml>"),
                TestData(1,"bigData",1.0,"<xml>Collokia is located in Uruguay.</xml>"),
                TestData(1,"bigData",1.0,"<xml>Stanford University is located in California. It is a great university.</xml>"),
                TestData(1,"bigData",1.0,"<xml>Stanford University is located in California. It is a great university.</xml>"))



        val testRdd = jsc.parallelize(test)

        val input = sparkSession.createDataFrame(testRdd, TestData::class.java)

        //println(input.collect())
        val coreNLP = CoreNLP(sparkSession, "pos, lemma, parse, ner").setInputCol("content")

        val parsed = coreNLP.transform(input)


        /*val ner = parsed.select("lemma").javaRDD().map { row ->
            val nerInDoc = row.get(0) as ArrayBuffer<ArrayBuffer<String>>


            for (nerInSentence in nerInDoc) {
                for (nerInToken in nerInSentence) {
                    print(nerInToken + " ")
                }
                println()
            }
            nerInDoc
        }
        ner.collect()*/
        parsed?.show()
        //val first = parsed.first().getAs[Row]("parsed")
    }

}

fun main(args: Array<String>) {


    val jsc = getLocalSparkContext("valami")
    val sparkSession = SparkSession.builder().master("local").appName("prediction").orCreate

    /*val test = LinkedList<Tuple2<Int,String>>()
    test.add(Tuple2(1, "<xml>Stanford University is located in California. It is a great university.</xml>"))
    test.add(Tuple2(2, "<xml>University of Szeged is located in Hungary. It is a great university.</xml>"))
    test.add(Tuple2(3, "<xml>Collokia is located in Uruguay.</xml>"))

    val testRdd = ctx.parallelizePairs(test).map{ item ->
        TestData(item._1,item._2)
    }

    testRdd.collect()*/

    val test = CoreNlpTest()
    test.testCoreNlp(jsc, sparkSession)


    closeSpark(jsc)

}






