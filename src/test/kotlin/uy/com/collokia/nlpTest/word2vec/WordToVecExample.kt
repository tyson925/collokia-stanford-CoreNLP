package uy.com.collokia.nlpTest.word2vec

import uy.com.collokia.common.utils.measureTimeInMillis
import uy.com.collokia.common.utils.rdd.closeSpark
import uy.com.collokia.common.utils.rdd.getLocalSparkContext

class WordToVecExample() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val time = measureTimeInMillis {
                val jsc = getLocalSparkContext("spark")

                val corpus = jsc.textFile("./../collokia-data-es-indexer/data/word2vec/spanish/text.txt").map { line ->
                    line.split(" ") as Iterable<String>
                }

                val model = org.apache.spark.mllib.feature.Word2Vec().fit(corpus)


                model.save(jsc.sc(), "./data/word2vec/spanish")
                model.transform("deporte")
                println(model.findSynonyms("deporte", 4).joinToString("\n"))

                closeSpark(jsc)
            }
            println("Excecution time was ${time.second}")

        }

    }


}


