package uy.com.collokia.nlpTest.word2vec

import uy.com.collokia.common.utils.rdd.getLocalSparkContext

class WordToVecExample() {
    companion object {
        @JvmStatic fun main(args: Array<String>) {
            val jsc = getLocalSparkContext("spark")

            val corpus = jsc.textFile("./../collokia-data-es-indexer/data/word2vec/spanish/text.txt").map { line ->
                line.split(" ") as Iterable<String>
            }

            val model = org.apache.spark.mllib.feature.Word2Vec().fit(corpus)
            model.transform("London")
            println(model.findSynonyms("London", 4))

        }

    }


}


