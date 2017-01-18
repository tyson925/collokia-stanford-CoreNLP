package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.tokenize.TokenizerModel
import opennlp.tools.tokenize.TokenizerThreadsafeME
import java.io.FileInputStream
import java.io.Serializable
import java.util.*

class OpenNlpTokenizerWrapper(private val modelName : String)  : Serializable {
    companion object {
        @Volatile @Transient private var tokenizerMap = HashMap<String, TokenizerThreadsafeME>()

        @Synchronized fun getTokenizer(modelName:String):TokenizerThreadsafeME{
            return tokenizerMap[modelName].let{
                it ?: TokenizerThreadsafeME(TokenizerModel(FileInputStream(modelName))).let{
                    tokenizerMap.put(modelName, it)
                    it
                }
            }
        }
    }

    @Synchronized fun get() : TokenizerThreadsafeME {

        return getTokenizer(modelName)
    }

}
