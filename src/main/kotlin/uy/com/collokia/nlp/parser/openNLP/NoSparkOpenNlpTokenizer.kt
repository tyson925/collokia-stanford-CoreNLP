@file:Suppress("unused", "ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.parser.openNLP

import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzedRaw
import uy.com.collokia.common.utils.nospark.NoSparkParamsDef
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1
import uy.com.collokia.nlp.parser.LANGUAGE
import java.io.Serializable


class NoSparkOpenNlpTokenizerRaw(val language: LANGUAGE = LANGUAGE.ENGLISH,
                                 sentenceDetectorModelName: String = englishSentenceDetectorModelName,
                                 tokenizerModelName: String = englishTokenizerModelName
) : NoSparkTransformer1to1<SimpleDocument,SimpleDocumentAnalyzed,String, List<String>>(
        SimpleDocument::content,
        SimpleDocumentAnalyzed::analyzedContent
), Serializable {
    val detectors = initDetectors(language, tokenizerModelName, sentenceDetectorModelName)

    override fun transfromData(dataIn: String): List<String> {
        val tokenizedText = try {
            detectors.first.get().sentDetect(dataIn).flatMap { sentence ->
                detectors.second.get().tokenize(sentence).toList()
            }
        } catch (e: Exception) {
            println("problem with content: " + dataIn)
            dataIn.split(Regex("\\W"))
        }
        return tokenizedText
    }
}


class NoSparkOpenNlpTokenizer(val language: LANGUAGE = LANGUAGE.ENGLISH,
                              sentenceDetectorModelName: String = englishSentenceDetectorModelName,
                              tokenizerModelName: String = englishTokenizerModelName
) : NoSparkTransformer1to1<SimpleDocument, SimpleDocumentAnalyzedRaw, String, List<List<String>>>(
        SimpleDocument::content,
        SimpleDocumentAnalyzedRaw::analyzedContent
), Serializable {

    val detectors = initDetectors(language, tokenizerModelName, sentenceDetectorModelName)

    override fun transfromData(dataIn: String): List<List<String>> {
        val tokenizedText = try {
            detectors.first.get().sentDetect(dataIn).map { sentence ->
                detectors.second.get().tokenize(sentence).toList()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            println("problem with content: " + dataIn)
            dataIn.split(".").map { sentence ->
                detectors.second.get().tokenize(sentence).toList()
            }
        }
        return tokenizedText
    }

}

private fun initDetectors(language: LANGUAGE, tokenizerModelName: String, sentenceDetectorModelName: String): Pair<OpenNlpSentenceDetectorWrapper, OpenNlpTokenizerWrapper> {
    var tokenizerWrapper: OpenNlpTokenizerWrapper = if (language == LANGUAGE.ENGLISH) {
        OpenNlpTokenizerWrapper(englishTokenizerModelName)
    } else if (language == LANGUAGE.SPANISH) {
        OpenNlpTokenizerWrapper(spanishTokenizerModelName)
    } else {
        OpenNlpTokenizerWrapper(tokenizerModelName)
    }
    var sdetectorWrapper: OpenNlpSentenceDetectorWrapper = if (language == LANGUAGE.ENGLISH) {
        OpenNlpSentenceDetectorWrapper(englishSentenceDetectorModelName)
    } else if (language == LANGUAGE.SPANISH) {
        OpenNlpSentenceDetectorWrapper(spanishSentenceDetectorModelName)
    } else {
        OpenNlpSentenceDetectorWrapper(sentenceDetectorModelName)
    }
    return Pair(sdetectorWrapper, tokenizerWrapper)
}

