package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.sentdetect.SentenceDetectorThreadsafeME
import opennlp.tools.sentdetect.SentenceModel
import java.io.FileInputStream
import java.io.Serializable

class OpenNlpSentenceDetectorWrapper(private val modelName: String) : Serializable {
    companion object {
        @Transient private var sDetector: SentenceDetectorThreadsafeME? = null
    }

    fun get(): SentenceDetectorThreadsafeME {

        return if (sDetector == null) {
            sDetector = SentenceDetectorThreadsafeME(SentenceModel(FileInputStream((modelName))))
            sDetector!!
        } else {
            sDetector!!
        }


    }

}