package uy.com.collokia.nlp.transformer.nospark

import org.apache.spark.ml.feature.RegexTokenizer
import scala.collection.JavaConversions
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzedBow
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1

class NoSparkRegexTokenizer(val regexTokenizer: RegexTokenizer) : NoSparkTransformer1to1<SimpleDocument, SimpleDocumentAnalyzedBow, String,List<String>>(
        SimpleDocument::content,
        SimpleDocumentAnalyzedBow::analyzedContent
) {

    private val transformFunc by lazy{ regexTokenizer.createTransformFunc() }

    override fun transfromData(dataIn:String): List<String> {
        return  JavaConversions.seqAsJavaList(transformFunc.apply(dataIn))

    }
}