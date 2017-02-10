package uy.com.collokia.nlp.transformer.nospark

import org.apache.spark.ml.feature.StopWordsRemover
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1

class NoSparkStopWordsRemover(_stopWordsRemover: StopWordsRemover) : NoSparkTransformer1to1<SimpleDocumentAnalyzed, SimpleDocumentAnalyzed, List<String>, List<String>>(
        SimpleDocumentAnalyzed::analyzedContent,
        SimpleDocumentAnalyzed::analyzedContent
) {
    override fun transfromData(dataIn: List<String>): List<String> {
        return stopWordsRemover.transform(dataIn)
    }

    class ExposedStopWordsRemover(_stopWordsRemover: StopWordsRemover) : StopWordsRemover() {
        init {
            _stopWordsRemover.params().forEach { p ->
                this.set(p.name(), _stopWordsRemover.get(p).get())
            }
        }

        val stopWordsSet by lazy { stopWords.toSet() }
        val lowerStopWordsSet by lazy { stopWords.map { it?.toLowerCase() }.toSet() }

        fun transform(dataIn: List<String>): List<String> {
            return if (caseSensitive) {
                dataIn.filter { t: String ->
                    !stopWordsSet.contains(t)
                }
            } else {
                // TODO: support user locale (SPARK-15064)
                dataIn.filter { t: String -> !lowerStopWordsSet.contains(t.toLowerCase()) }
            }
        }
    }
    val stopWordsRemover = ExposedStopWordsRemover(_stopWordsRemover)

}