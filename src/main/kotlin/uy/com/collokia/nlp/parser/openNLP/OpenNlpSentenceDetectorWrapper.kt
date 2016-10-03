package uy.com.collokia.nlp.parser.openNLP

import opennlp.tools.sentdetect.SentenceDetectorME
import opennlp.tools.sentdetect.SentenceModel
import java.io.FileInputStream
import java.io.Serializable

class OpenNlpSentenceDetectorWrapper(private val modelName: String) : Serializable {
    @Transient private var sDetector: SentenceDetectorME? = null

    fun get(): SentenceDetectorME {

        return if (sDetector == null) {
            sDetector = SentenceDetectorME(SentenceModel(FileInputStream((modelName))))
            sDetector!!
        } else {
            sDetector!!
        }


    }

}