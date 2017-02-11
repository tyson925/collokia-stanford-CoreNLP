package uy.com.collokia.nlp.transformer.vectorTransformer

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentExtractedFeatures
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1

class SimpleDocumentExtractedFeaturesAsVector(
        id: String ="",
        title: String="",
        url: String="",
        content: String="",
        tags: List<String> =listOf(),
        category: String ="",
        var featureVector: Vector = SparseVector(0, IntArray(0), DoubleArray(0))) : SimpleDocument(
        id,
        title,
        url,
        content,
        tags,
        category
)


class NoSparkVectorUnaryTransformer :NoSparkTransformer1to1<SimpleDocumentExtractedFeatures, SimpleDocumentExtractedFeaturesAsVector, DoubleArray, Vector>(
    SimpleDocumentExtractedFeatures::extractedFeatures,
        SimpleDocumentExtractedFeaturesAsVector::featureVector
) {
    override fun transfromData(dataIn: DoubleArray): Vector {
        return Vectors.dense(dataIn)
    }
}

