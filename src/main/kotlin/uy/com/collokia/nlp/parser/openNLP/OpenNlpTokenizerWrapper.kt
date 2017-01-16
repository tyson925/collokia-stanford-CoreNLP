package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.tokenize.TokenizerModel
import opennlp.tools.tokenize.TokenizerThreadsafeME
import java.io.FileInputStream
import java.io.Serializable

class OpenNlpTokenizerWrapper(private val modelName : String)  : Serializable {
    companion object {
        @Transient private var tokenizer: TokenizerThreadsafeME? = null
    }

    fun get() : TokenizerThreadsafeME {

        return if (tokenizer == null) {
            tokenizer = TokenizerThreadsafeME(TokenizerModel(FileInputStream(modelName)))
            tokenizer!!
        } else {
            tokenizer!!
        }
    }

}
