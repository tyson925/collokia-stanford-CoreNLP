package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument

import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzedBow
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1

class SimpleDocumentNgramsBow(
        id: String ="",
        title: String="",
        url: String="",
        content: String="",
        tags: List<String> =listOf(),
        category: String ="",
        var ngrams: List<String> = listOf()
) : SimpleDocument(
        id,
        title,
        url,
        content,
        tags,
        category
)


class NoSparkNGramBowInput(val nGramBowInput: NGramInRawInput) : NoSparkTransformer1to1<SimpleDocumentAnalyzedBow, SimpleDocumentNgramsBow, List<String>, List<String>>(
        SimpleDocumentAnalyzedBow::analyzedContent,
        SimpleDocumentNgramsBow::ngrams
) {
    override fun transfromData(dataIn: List<String>): List<String>
            = transformFunc.apply(WrappedArray.make<String>(dataIn.toTypedArray())).toList()

    val transformFunc = nGramBowInput.createTransformFunc()


    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkNGramBowInput>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkNGramBowInput")

}




