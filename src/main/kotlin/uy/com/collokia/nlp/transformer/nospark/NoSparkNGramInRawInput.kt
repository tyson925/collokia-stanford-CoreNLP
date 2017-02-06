package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.*
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.nospark.NoSparkTransformer
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer



class NoSparkNGramInRawInput(val nGramRawInput: NGramInRawInput) : NoSparkTransformer(){

    override fun transformSchema(schema: StructType): StructType {
        //add the output field to the schema
        val inputType = schema.apply(nGramRawInput.inputCol).dataType()
        if (!inputType.sameType(DataTypes.createArrayType(DataTypes.StringType))){
            throw  IllegalArgumentException("Input column ${nGramRawInput.inputCol} should be of type Array<String>.")
        }

        if (schema.fieldNames().contains(nGramRawInput.outputCol)) {
            throw  IllegalArgumentException("Output column ${nGramRawInput.outputCol} already exists.")
        }
        val outputFields = schema.fields() +
                StructField(nGramRawInput.outputCol, DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
        return StructType(outputFields)
    }


    val transformFunc = nGramRawInput.createTransformFunc()


    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        return mapIn +  (nGramRawInput.outputCol to WrappedArray.make<String>(transformFunc.apply(mapIn[nGramRawInput.inputCol] as WrappedArray<String>)))
    }

    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkNGramInRawInput>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkNGramInRawInput")

}




