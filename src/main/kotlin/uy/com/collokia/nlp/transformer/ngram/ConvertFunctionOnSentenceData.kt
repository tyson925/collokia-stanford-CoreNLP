package uy.com.collokia.nlp.transformer.ngram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.parser.DEFAULT_NGRAM_SEPARATOR
import uy.com.collokia.nlp.parser.NLPToken
import java.io.Serializable
import java.util.*


@Suppress("unused")
class ConvertFunctionOnSentenceData :
        AbstractFunction1<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>, Array<Array<Map<String, String>>>>(), Serializable {

    companion object {
        const val NUMBER_OF_NGRAMS = 3
    }

    override fun apply(content: WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>)
            : Array<Array<Map<String, String>>> {

        val sentences = JavaConversions.seqAsJavaList(content)
        //println(sentences.javaClass.kotlin)
        return sentences.map { sentence ->
            //println(sentence.javaClass.kotlin)
            wordNGrams(sentence, NUMBER_OF_NGRAMS, true, separator = DEFAULT_NGRAM_SEPARATOR)
        }.toTypedArray()

    }

    fun wordNGrams(sentence: WrappedArray<scala.collection.immutable.Map<String, String>>,
                   N: Int,
                   oneToN: Boolean,
                   separator: String = DEFAULT_NGRAM_SEPARATOR)
            : Array<Map<String, String>> {

        val RET = LinkedList<Map<String, String>>()

        for (i in (if (oneToN) 1 else N)..N + 1 - 1) {
            RET.addAll(wordNGramsLevel(JavaConversions.seqAsJavaList(sentence), i, separator))
        }

        return RET.toTypedArray()
    }


    /**
     * @param tokensInfo
     * *
     * @param N
     * *
     * @return
     */
    private fun wordNGramsLevel(tokensInfo: List<scala.collection.immutable.Map<String, String>>,
                                N: Int,
                                separator: String = DEFAULT_NGRAM_SEPARATOR)
            : List<Map<String, String>> {

        val RET: List<Map<String, String>>

        if (N < 2) {
            RET = listOf(tokensInfo.map { map ->
                val item = map.iterator().next()
                item._1 to item._2
            }.toMap())
        } else {
            RET = mutableListOf<Map<String, String>>()
            for (i in 0..tokensInfo.size - N + 1 - 1) {
                val tokens = StringBuffer()
                val lemmas = StringBuffer()
                //val posTags = StringBuffer()
                val firstTokenMap = tokensInfo[i]
                for (j in 0..N - 1) {
                    val tokenMap = tokensInfo[i + j]
                    tokens.append(tokenMap[NLPToken::token.name].get())
                    lemmas.append(tokenMap[NLPToken::lemma.name].get())
                    //posTags.append(tokensInfo[i + j].apply(2))
                    if (j < (N - 1)) {
                        tokens.append(separator)
                        lemmas.append(separator)
                        //posTags.append(separator)
                    }
                }

                RET.add(mapOf(NLPToken::token.name to tokens.toString(),
                        NLPToken::lemma.name to lemmas.toString(),
                        NLPToken::index.name to firstTokenMap[NLPToken::index.name].get(),
                        NLPToken::indexInContent.name to firstTokenMap[NLPToken::indexInContent.name].get()
                ))
            }
        }

        return RET
    }
}
