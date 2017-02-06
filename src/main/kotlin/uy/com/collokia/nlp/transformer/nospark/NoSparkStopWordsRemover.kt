package uy.com.collokia.nlp.transformer.nospark

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import scala.Function1
import scala.collection.Seq
import uy.com.collokia.common.utils.nospark.NoSparkTransformer

class NoSparkStopWordsRemover(_stopWordsRemover: StopWordsRemover) : NoSparkTransformer() {

    override fun transformSchema(schema: StructType): StructType {
        //add the output field to the schema
        val inputType = schema.apply(stopWordsRemover.inputCol).dataType()
        if (!inputType.sameType(DataTypes.createArrayType(DataTypes.StringType))){
            throw  IllegalArgumentException("Input column ${stopWordsRemover.inputCol} should be of type Array<String>.")
        }

        if (schema.fieldNames().contains(stopWordsRemover.outputCol)) {
            throw  IllegalArgumentException("Output column ${stopWordsRemover.outputCol} already exists.")
        }
        val outputFields = schema.fields() +
                StructField(stopWordsRemover.outputCol, DataTypes.createArrayType(DataTypes.StringType), false, org.apache.spark.sql.types.Metadata.empty())
        return StructType(outputFields)
    }

    override fun copy(extra: ParamMap?): Transformer = defaultCopy<NoSparkRegexTokenizer>(extra)


    override fun uid(): String = org.apache.spark.ml.util.`Identifiable$`.`MODULE$`.randomUID("NoSparkStopWordsRemover")

    class ExposedStopWordsRemover(_stopWordsRemover: StopWordsRemover) : StopWordsRemover() {
        init {
            _stopWordsRemover.params().forEach { p ->
                this.set(p.name(), _stopWordsRemover.get(p).get())
            }
        }

        val stopWordsSet by lazy { stopWords.toSet() }
        val lowerStopWordsSet by lazy { stopWords.map { it?.toLowerCase() }.toSet() }

        fun transform(dataIn: Collection<String>): Collection<String> = if (caseSensitive) {
            dataIn.filter { !stopWordsSet.contains(it) }
        } else {
            // TODO: support user locale (SPARK-15064)
            dataIn.filter { !lowerStopWordsSet.contains(it.toLowerCase()) }
        }


    }

    val stopWordsRemover = ExposedStopWordsRemover(_stopWordsRemover)


    override fun transfromRow(mapIn: Map<String, Any>): Map<String, Any> {
        return mapIn + (stopWordsRemover.outputCol to stopWordsRemover.transform(mapIn[stopWordsRemover.inputCol] as Collection<String>))
    }
}