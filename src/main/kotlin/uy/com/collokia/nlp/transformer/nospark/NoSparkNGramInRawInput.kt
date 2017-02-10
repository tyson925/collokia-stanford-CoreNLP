package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.*
import scala.collection.mutable.WrappedArray

import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentNgrams
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1


class NoSparkNGramInRawInput(val nGramRawInput: NGramInRawInput) : NoSparkTransformer1to1<SimpleDocumentAnalyzed, SimpleDocumentNgrams, List<String>, List<List<Map<String,String>>>>(
        SimpleDocumentAnalyzed::analyzedContent,
        SimpleDocumentNgrams::analyzedContent
) {
    override fun transfromData(dataIn: List<String>): List<List<Map<String, String>>> {

    }



    val transformFunc = nGramRawInput.createTransformFunc()


    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        return mapIn +  (nGramRawInput.outputCol to WrappedArray.make<String>(transformFunc.apply(mapIn[nGramRawInput.inputCol] as WrappedArray<String>)))
    }

    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkNGramInRawInput>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkNGramInRawInput")

}




