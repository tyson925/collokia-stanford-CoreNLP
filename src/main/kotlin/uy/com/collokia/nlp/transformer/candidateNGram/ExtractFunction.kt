package uy.com.collokia.nlp.transformer.candidateNGram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.transformer.ngram.number_of_grams
import java.io.Serializable
import java.util.*

class ExtractFunction : AbstractFunction1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>>, Serializable {

    val language: LANGUAGE

    constructor(language: LANGUAGE) {
        this.language = language
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
            if (language == LANGUAGE.ENGLISH) {
                RET = tokens.filter { token -> filterEnglishCandidate(listOf(token.apply(2))) }.map { token -> token.apply(1) }.toMutableList()
            } else {
                RET = tokens.filter { token -> filterSpanishCandidate(listOf(token.apply(2)), listOf(token.apply(1))) }.map { token ->
                    token.apply(1)
                }.toMutableList()
            }

        } else {
            RET = mutableListOf<String>()
            for (i in 0..tokens.size - N + 1 - 1) {
                val candidateTokens = StringBuffer()
                val candidatePOSs = mutableListOf<String>()
                val candidateLemmas = mutableListOf<String>()
                for (j in 0..N - 1) {
                    candidateTokens.append(tokens[i + j].apply(1))
                    candidateLemmas.add(tokens[i + j].apply(1))
                    candidatePOSs.add(tokens[i + j].apply(2))
                    if (j < (N - 1)) {
                        candidateTokens.append(separator)
                    }
                }
                if (language == LANGUAGE.ENGLISH) {
                    if (filterEnglishCandidate(candidatePOSs)) {
                        RET.add(candidateTokens.toString())
                    }
                } else {
                    if (filterSpanishCandidate(candidatePOSs, candidateLemmas)) {
                        RET.add(candidateTokens.toString())
                    }
                }
            }
        }

        return RET
    }

    private fun filterEnglishCandidate(candidatePOSs: List<String>): Boolean {
        val filteredPOSs = candidatePOSs.filter { pos -> if (pos.startsWith("NN") || pos == "JJ" || pos.equals(Regex("VB*"))) true else false }
        return if (filteredPOSs.size == candidatePOSs.size) true else false
    }

    private fun filterSpanishCandidate(candidatePOSs: List<String>, candidateLemma: List<String>): Boolean {


        val filteredPOSs = candidatePOSs.filterIndexed { index, pos ->
            if (pos == "n" || (pos == "v" && candidateLemma[index] != "ser")) true else false
        }

        return when {
            (filteredPOSs.size == candidatePOSs.size) -> true
            else -> false
        }
    }

}
