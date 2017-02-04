package uy.com.collokia.nlp.transformer.ngram

import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import scala.runtime.AbstractFunction1
import uy.com.collokia.nlp.parser.DEFAULT_NGRAM_SEPARATOR
import java.io.Serializable
import java.util.*


class ConvertFunctionOnRawData : AbstractFunction1<WrappedArray<String>, Array<String>>, Serializable {


    val NGRAM_SEPARATOR : String

    constructor(ngram_separator : String = DEFAULT_NGRAM_SEPARATOR){
        this.NGRAM_SEPARATOR = ngram_separator
    }

    companion object {

        const val NUMBER_OF_NGRAMS = 3
    }

    override fun apply(content: WrappedArray<String>?): Array<String> {

        return wordNGrams(JavaConversions.seqAsJavaList(content) ?: listOf(), NUMBER_OF_NGRAMS, true, NGRAM_SEPARATOR).toTypedArray()
    }

    fun wordNGrams(tokens: List<String>, N: Int, oneToN: Boolean, separator: String): List<String> {
        val RET = LinkedList<String>()

        for (i in (if (oneToN) 1 else N)..N + 1 - 1) {
            RET.addAll(wordNGramsLevel(tokens, i, separator))
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
    private fun wordNGramsLevel(tokens: List<String>, N: Int, separator: String): List<String> {
        val RET: MutableList<String>

        if (N < 2) {
            RET = tokens.toMutableList()
        } else {
            RET = mutableListOf<String>()
            for (i in 0..tokens.size - N + 1 - 1) {
                val buf = StringBuffer()
                for (j in 0..N - 1) {
                    buf.append(tokens[i + j])
                    if (j < (N - 1)) {
                        buf.append(separator)
                    }
                }
                RET.add(buf.toString())
            }
        }

        return RET
    }
}