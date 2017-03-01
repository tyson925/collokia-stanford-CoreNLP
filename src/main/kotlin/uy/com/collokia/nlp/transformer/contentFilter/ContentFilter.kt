@file:Suppress("unused")

package uy.com.collokia.nlp.transformer.contentFilter

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.functions
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.parser.openNLP.TOKENIZED_CONTENT_COL_NAME


const val FILTERED_CONTENT_COL_NAME = "filteredContent"

class ContentFilter : Transformer {


    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val potentialTags: List<String>

    constructor(potentialTags: List<String>,
                sparkSession: SparkSession,
                inputColName: String = TOKENIZED_CONTENT_COL_NAME,
                outputColName: String = FILTERED_CONTENT_COL_NAME) {

        this.potentialTags = potentialTags
        this.sparkSession = sparkSession
        this.inputColName = inputColName
        this.outputColName = outputColName

        val contentFilter = UDF1({ tokens: WrappedArray<String> ->

            this.potentialTags.intersect(JavaConversions.asJavaCollection(tokens))

        })

        this.sparkSession.udf().register("contentFilter", contentFilter, DataTypes.createArrayType(DataTypes.StringType))
    }

    fun setInputColName(inputColName: String): ContentFilter {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): ContentFilter {
        this.outputColName = outputColName
        return this
    }


    override fun uid(): String {
        return "contentFilter11111111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return ContentFilter(this.potentialTags, this.sparkSession, this.inputColName, this.outputColName)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row> {
        return dataset?.let {
            dataset.select(dataset.col("*"),
                    functions.callUDF("contentFilter", JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
        } ?: sparkSession.emptyDataFrame()
    }

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        //val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes) {
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.StringType), inputType?.nullable() ?: false)
    }
}
