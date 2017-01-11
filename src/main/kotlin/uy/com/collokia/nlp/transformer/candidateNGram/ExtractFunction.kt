package uy.com.collokia.nlp.transformer.candidateNGram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.transformer.ngram.number_of_grams
import java.io.Serializable
import java.util.*

class ExtractFunction : AbstractFunction1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>>, Serializable {



    constructor(){

    }

    override fun apply(tokens: WrappedArray<WrappedArray<WrappedArray<String>>>): Array<String> {
        val sentences = JavaConversions.seqAsJavaList(tokens)
        //println(sentences.javaClass.kotlin)
        return sentences.flatMap { sentence ->
            //println(sentence.javaClass.kotlin)
            wordNGrams(sentence, number_of_grams, true, " ")
        }.toTypedArray()

    }


    fun wordNGrams(sentence: WrappedArray<WrappedArray<String>>, N: Int, oneToN: Boolean, separator: String = "_"): List<String> {
        val RET = LinkedList<String>()

        for (i in (if (oneToN) 1 else N)..N + 1 - 1) {
            RET.addAll(wordNGramsLevel(JavaConversions.seqAsJavaList(sentence), i, separator))
        }

        return RET
    }

    /**
     * @param tokens
     * *
     * @param N
     * *
     * @return
     */
    private fun wordNGramsLevel(tokens: List<WrappedArray<String>>, N: Int, separator: String = "_"): List<String> {
        val RET: MutableList<String>

        if (N < 2) {
            RET = tokens.filter { token -> filterCandidate(listOf(token.apply(2))) }.map { token -> token.apply(1) }.toMutableList()
            /*val candidatePOSs = tokens.map { token -> token.apply(2) }
            if (filterCandidate(candidatePOSs)) {
                RET = tokens.map { token -> token.apply(1) }.toMutableList()
            } else {
                RET = mutableListOf<String>()
            }*/
        } else {
            RET = mutableListOf<String>()
            for (i in 0..tokens.size - N + 1 - 1) {
                val candidateTokens = StringBuffer()
                val candidatePOSs = mutableListOf<String>()
                for (j in 0..N - 1) {
                    candidateTokens.append(tokens[i + j].apply(1))
                    candidatePOSs.add(tokens[i + j].apply(2))
                    if (j < (N - 1)) {
                        candidateTokens.append(separator)
                    }
                }
                if (filterCandidate(candidatePOSs)) {
                    RET.add(candidateTokens.toString())
                }
            }
        }

        return RET
    }

    private fun filterCandidate(candidatePOSs: List<String>): Boolean {
        val filteredPOSs = candidatePOSs.filter { pos -> if (pos.startsWith("NN") || pos == "ADJ" || pos.startsWith("V")) true else false }
        return if (filteredPOSs.size == candidatePOSs.size) true else false
    }

}
