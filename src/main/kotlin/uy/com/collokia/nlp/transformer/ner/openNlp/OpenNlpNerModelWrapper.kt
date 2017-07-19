package uy.com.collokia.nlp.transformer.ner.openNlp

import opennlp.tools.cmdline.namefind.TokenNameFinderModelLoader
import opennlp.tools.namefind.TokenNameFinderModel
import java.io.File
import java.io.Serializable
import java.util.*

class OpenNlpNerModelWrapper(private val modelName : String)  : Serializable {
    companion object {
        @Volatile @Transient private var nerMap = HashMap<String, TokenNameFinderModel>()

        @Synchronized fun getNerModel(modelName:String): TokenNameFinderModel {
            return nerMap[modelName].let{

                it ?: TokenNameFinderModelLoader().load(File(modelName)).let{
                    nerMap.put(modelName, it)
                    it
                }
            }
        }
    }

    @Synchronized fun get() : TokenNameFinderModel {

        return getNerModel(modelName)
    }

}
