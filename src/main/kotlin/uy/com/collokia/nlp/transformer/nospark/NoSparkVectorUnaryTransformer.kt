package uy.com.collokia.nlp.transformer.vectorTransformer

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.*
import scala.Function1
import scala.collection.mutable.WrappedArray
import scala.reflect.`ClassTag$`
import scala.runtime.AbstractFunction1
import uy.com.collokia.common.utils.nospark.NoSparkTransformer
import uy.com.collokia.nlp.transformer.nospark.NoSparkRegexTokenizer
import java.io.Serializable



class NoSparkVectorUnaryTransformer(val vectorUnaryTransformer:VectorUnaryTransformer) :NoSparkTransformer(){

    private val transformFunc = vectorUnaryTransformer.createTransformFunc()!!

    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        return mapIn +  (vectorUnaryTransformer.outputCol to transformFunc.apply(mapIn[vectorUnaryTransformer.inputCol] as WrappedArray<Double>))
    }

    override fun transformSchema(schema: StructType): StructType {
        //add the output field to the schema
        val inputType = schema.apply(vectorUnaryTransformer.inputCol).dataType()
        if (!inputType.sameType(DataTypes.createArrayType(DataTypes.DoubleType))){
            throw  IllegalArgumentException("Input column ${vectorUnaryTransformer.inputCol} should be of type Array<Double>.")
        }

        if (schema.fieldNames().contains(vectorUnaryTransformer.outputCol)) {
            throw  IllegalArgumentException("Output column ${vectorUnaryTransformer.outputCol} already exists.")
        }
        val outputFields = schema.fields() +
                StructField(vectorUnaryTransformer.outputCol, VectorUDT(), false, Metadata.empty())
        return StructType(outputFields)
    }

    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkVectorUnaryTransformer>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkVectorUnaryTransformer")
}

