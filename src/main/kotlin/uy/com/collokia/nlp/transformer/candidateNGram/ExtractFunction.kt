package uy.com.collokia.nlp.transformer.candidateNGram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.parser.DEFAULT_NGRAM_SEPARATOR
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.NLPToken
import uy.com.collokia.nlp.parser.PosToken
import java.io.Serializable
import java.util.*

class ExtractFunction :
        AbstractFunction1<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>,
                Array<String>>, Serializable {

    companion object {
        const val NUMBER_OF_NGRAMS = 4
    }

    val language: LANGUAGE

    constructor(language: LANGUAGE) {
        this.language = language
    }

    override fun apply(content: WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>)
            : Array<String> {
        val sentences = JavaConversions.seqAsJavaList(content)
        //println(sentences.javaClass.kotlin)
        return sentences.flatMap { sentence ->
            //println(sentence.javaClass.kotlin)
            val res = wordNGrams(sentence, NUMBER_OF_NGRAMS, true, separator = DEFAULT_NGRAM_SEPARATOR)
            res.asIterable()
        }.toTypedArray()

    }


    fun wordNGrams(sentence: WrappedArray<scala.collection.immutable.Map<String, String>>,
                   N: Int,
                   oneToN: Boolean,
                   separator: String = "_"): Array<String> {

        val RET = LinkedList<String>()

        for (i in (if (oneToN) 1 else N)..N + 1 - 1) {
            RET.addAll(wordNGramsLevel(JavaConversions.seqAsJavaList(sentence), i, separator))
        }

        return RET.toTypedArray()
    }

    /**
     * @param tokens
     * *
     * @param N
     * *
     * @return
     */
    private fun wordNGramsLevel(tokens: List<scala.collection.immutable.Map<String, String>>,
                                N: Int,
                                separator: String = "_")
            : List<String> {

        val RET: List<String>

        if (N < 2) {
            if (language == LANGUAGE.ENGLISH) {
                RET = tokens.filter { token ->
                    filterEnglishCandidate(listOf(token[PosToken::posTag.name].get()))
                }.map { token -> token[NLPToken::token.name].get() }.toMutableList()
            } else {
                RET = tokens.filter { token ->
                    filterSpanishCandidate(listOf(token[PosToken::posTag.name].get()), listOf(token[NLPToken::token.name].get()))
                }.map { token ->
                    getSpanishLemma(token)
                }.toMutableList()
            }

        } else {
            RET = mutableListOf<String>()
            for (i in 0..tokens.size - N + 1 - 1) {
                val candidateTokens = StringBuffer()
                val candidatePOSs = mutableListOf<String>()
                val candidateLemmas = mutableListOf<String>()
                for (j in 0..N - 1) {
                    if (language == LANGUAGE.ENGLISH) {
                        candidateTokens.append(tokens[i + j][NLPToken::lemma.name].get())
                    } else {
                        candidateTokens.append(getSpanishLemma(tokens[i + j]))
                    }

                    candidateLemmas.add(tokens[i + j][NLPToken::lemma.name].get())
                    candidatePOSs.add(tokens[i + j][PosToken::posTag.name].get())
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

        val filteredPOSs = candidatePOSs.filter { pos -> pos.startsWith("NN") || pos == "JJ" || pos.equals(Regex("VB*")) }
        return filteredPOSs.size == candidatePOSs.size
    }

    private fun filterSpanishCandidate(candidatePOSs: List<String>, candidateLemma: List<String>): Boolean {

        val filteredPOSs = if (candidatePOSs.size > 2) {
            if (isValidSequence(candidatePOSs, candidateLemma)) candidatePOSs else listOf()
        } else {
            candidatePOSs.filterIndexed { index, pos ->
                isValidToken(pos, candidateLemma[index])
            }
        }

        return filteredPOSs.size == candidatePOSs.size

    }

    private fun isValidSequence(candidatePOSs: List<String>, candidateLemma: List<String>): Boolean {
        val isValidFistToken = isValidToken(candidatePOSs.first(), candidateLemma.first())
        val isValidLastToken = isValidToken(candidatePOSs.last(), candidateLemma.last())

        val middleTokens = (1..candidatePOSs.size-2).map { index ->
            isValidToken(candidatePOSs[index],candidateLemma[1]) || isSpanishADP(candidatePOSs[index])
        }.filter { it }

        val isValidSequence = middleTokens.size == candidatePOSs.size -2

        return isValidSequence && isValidFistToken && isValidLastToken

    }

    private fun isValidToken(pos: String, lemma: String): Boolean {
        return isSpanishNoun(pos) || isSpanishVerb(pos, lemma)
    }

    private fun isSpanishVerb(pos: String, verb: String): Boolean {
        return (pos == "VERB" && (verb != "ser" && verb != "esta" &&
                verb != "estar" && verb != "ir"
                && verb != "del"))

    }

    private fun isSpanishNoun(pos: String): Boolean {
        return pos == "PROPN" || pos == "NOUN"
    }

    private fun isSpanishADP(pos: String): Boolean {
        return pos == "ADP"
    }

    private fun getSpanishLemma(token: scala.collection.immutable.Map<String, String>): String {
        return if (token[PosToken::posTag.name].get() == "NOUN" || token[PosToken::posTag.name].get() == "PROPN") {
            token[NLPToken::token.name].get()
        } else {
            token[NLPToken::lemma.name].get()
        }
    }

}
