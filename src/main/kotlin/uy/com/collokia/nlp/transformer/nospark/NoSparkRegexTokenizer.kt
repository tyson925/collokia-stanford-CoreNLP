package uy.com.collokia.nlp.transformer.nospark

import org.apache.spark.ml.feature.RegexTokenizer
import scala.collection.JavaConversions
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1

class NoSparkRegexTokenizer(val regexTokenizer: RegexTokenizer) : NoSparkTransformer1to1<SimpleDocument, SimpleDocumentAnalyzed, String,List<String>>(
        SimpleDocument::content,
        SimpleDocumentAnalyzed::analyzedContent
) {

    private val transformFunc by lazy{ regexTokenizer.createTransformFunc() }

    override fun transfromData(dataIn:String): List<String> {
        return  JavaConversions.seqAsJavaList(transformFunc.apply(dataIn))

    }
}