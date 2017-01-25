package uy.com.collokia.nlp.transformer.ngram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.parser.DEFAULT_NGRAM_SEPARATOR
import java.io.Serializable
import java.util.*



@Suppress("unused")
class ConvertFunctionOnSentenceData : AbstractFunction1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<Array<Array<String>>>>(), Serializable {

    companion object {
        const val NUMBER_OF_NGRAMS = 3
    }

    override fun apply(content: WrappedArray<WrappedArray<WrappedArray<String>>>): Array<Array<Array<String>>> {
        val sentences = JavaConversions.seqAsJavaList(content)
        //println(sentences.javaClass.kotlin)
        return sentences.map { sentence ->
            //println(sentence.javaClass.kotlin)
            wordNGrams(sentence, NUMBER_OF_NGRAMS, true, separator = DEFAULT_NGRAM_SEPARATOR)
        }.toTypedArray()

    }

    fun wordNGrams(sentence: WrappedArray<WrappedArray<String>>, N: Int, oneToN: Boolean, separator: String = DEFAULT_NGRAM_SEPARATOR): Array<Array<String>> {
        val RET = LinkedList<Array<String>>()

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
    private fun wordNGramsLevel(tokensInfo: List<WrappedArray<String>>, N: Int, separator: String = DEFAULT_NGRAM_SEPARATOR): List<Array<String>> {
        val RET: MutableList<Array<String>>

        if (N < 2) {
            RET = tokensInfo.map { token -> JavaConversions.seqAsJavaList(token).toTypedArray() }.toMutableList()
        } else {
            RET = mutableListOf<Array<String>>()
            for (i in 0..tokensInfo.size - N + 1 - 1) {
                val tokens = StringBuffer()
                val lemmas = StringBuffer()
                //val posTags = StringBuffer()

                for (j in 0..N - 1) {
                    tokens.append(tokensInfo[i + j].apply(0))
                    lemmas.append(tokensInfo[i + j].apply(1))
                    //posTags.append(tokensInfo[i + j].apply(2))
                    if (j < (N - 1)) {
                        tokens.append(separator)
                        lemmas.append(separator)
                        //posTags.append(separator)
                    }
                }

                RET.add(arrayOf(tokens.toString(),lemmas.toString()))
            }
        }

        return RET
    }
}
