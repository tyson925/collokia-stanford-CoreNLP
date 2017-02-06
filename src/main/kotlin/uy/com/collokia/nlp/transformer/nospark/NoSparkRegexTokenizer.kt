package uy.com.collokia.nlp.transformer.nospark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.Function1
import scala.collection.JavaConversions
import scala.collection.Seq
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.utils.nospark.NoSparkTransformer

class NoSparkRegexTokenizer(val regexTokenizer: RegexTokenizer) : NoSparkTransformer() {

    override fun transformSchema(schema: StructType): StructType {
        //add the output field to the schema
        //val inputType = schema.apply(_inputCol).dataType()
        if (schema.fieldNames().contains(regexTokenizer.outputCol)) {
            throw  IllegalArgumentException("Output column ${regexTokenizer.outputCol} already exists.")
        }
        val outputFields = schema.fields() +
                StructField(regexTokenizer.outputCol, DataTypes.createArrayType(DataTypes.StringType), false, org.apache.spark.sql.types.Metadata.empty())
        return StructType(outputFields)
    }

    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkRegexTokenizer>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkStopWordsRemover")


    private val transformFunc = regexTokenizer.createTransformFunc()

    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        return mapIn + (regexTokenizer.outputCol to
                WrappedArray.make<String>(JavaConversions.seqAsJavaList(
                        transformFunc.apply(mapIn[regexTokenizer.inputCol].toString())
                ).toTypedArray()
                ))
    }

}