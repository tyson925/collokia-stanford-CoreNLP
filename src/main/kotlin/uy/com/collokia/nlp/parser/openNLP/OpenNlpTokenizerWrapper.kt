package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.tokenize.TokenizerME
import opennlp.tools.tokenize.TokenizerModel
import java.io.FileInputStream
import java.io.Serializable

class OpenNlpTokenizerWrapper(private val modelName : String)  : Serializable {
    //companion object {
        @Transient private var tokenizer: TokenizerME? = null
    //}

    fun get() : TokenizerME {

        return if (tokenizer == null) {
            tokenizer = TokenizerME(TokenizerModel(FileInputStream(modelName)))
            tokenizer!!
        } else {
            tokenizer!!
        }
    }

}
