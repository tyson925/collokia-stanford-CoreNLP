package uy.com.collokia.nlp.transformer.vectorTransformer

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.DataType
import scala.Function1
import scala.collection.mutable.WrappedArray
import scala.reflect.`ClassTag$`
import scala.runtime.AbstractFunction1
import java.io.Serializable

class VectorUnaryTransformer() : UnaryTransformer<WrappedArray<Double>, Vector, VectorUnaryTransformer>() {

    override fun uid(): String? {
        return "uid1111111"
    }

    override fun outputDataType(): DataType? {
        return VectorUDT()
    }

    override fun createTransformFunc(): Function1<WrappedArray<Double>, Vector>? {
        return ConvertFunction()
    }

    class ConvertFunction : AbstractFunction1<WrappedArray<Double>, Vector>(), Serializable {
        override fun apply(tokens: WrappedArray<Double>?): Vector? {
            return Vectors.dense(tokens?.toVector()?.toArray(`ClassTag$`.`MODULE$`.Double()) as DoubleArray)
        }
    }
}

