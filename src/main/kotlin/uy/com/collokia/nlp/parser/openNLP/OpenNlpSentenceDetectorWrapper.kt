package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.sentdetect.SentenceDetectorME
import opennlp.tools.sentdetect.SentenceModel
import java.io.FileInputStream
import java.io.Serializable
import java.util.*

class OpenNlpSentenceDetectorWrapper(private val modelName: String) : Serializable {
    companion object {
        @Volatile @Transient private var modelMap = HashMap<String,SentenceModel>()

        @Synchronized fun getModel( modelName:String):SentenceModel{
            return modelMap[modelName].let{
                it ?: SentenceModel(FileInputStream(modelName)).let{
                    modelMap.put(modelName, it)
                    it
                }
            }
        }
    }

    fun get(): SentenceDetectorME {
            return SentenceDetectorME(getModel(modelName))
    }

}