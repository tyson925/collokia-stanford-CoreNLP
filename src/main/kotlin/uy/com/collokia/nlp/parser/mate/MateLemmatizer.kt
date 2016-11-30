@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate

import is2.data.SentenceData09
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent

const val lemmatizedContentCol = "lemmatizedContent"

class MateLemmatizer : Transformer {

    val lemmatizerWrapper: LematizerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val isRaw: Boolean


    constructor(sparkSession: SparkSession,
                isRaw: Boolean,
                lemmatizerModel: String = "./../../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model",
                inputColName: String = tokenizedContent,
                outputColName: String = lemmatizedContentCol) {

        this.sparkSession = sparkSession
        this.isRaw = isRaw

        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        this.inputColName = inputColName
        this.outputColName = outputColName

        val lemmatizer = org.apache.spark.sql.api.java.UDF1({ tokens: scala.collection.mutable.WrappedArray<String> ->

            val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

            val lemmatized = SentenceData09()
            lemmatized.init(sentenceArray)

            if (isRaw) {
                lemmatizerWrapper.get().apply(lemmatized).plemmas.joinToString(" ")
            } else {
                lemmatizerWrapper.get().apply(lemmatized).plemmas.toList()
            }
        })

        if (isRaw) {
            sparkSession.udf().register("lemmatizer", lemmatizer, DataTypes.StringType)
        } else {
            sparkSession.udf().register("lemmatizer", lemmatizer, DataTypes.createArrayType(DataTypes.StringType))
        }

    }

    fun setInputColName(inputColName: String): MateLemmatizer {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): MateLemmatizer {
        this.outputColName = outputColName
        return this
    }

    override fun uid(): String {
        return "lemmatizer111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateLemmatizer(this.sparkSession, this.isRaw, this.inputColName, this.outputColName)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        val outputDataType = transformSchema(dataset?.schema()).apply(outputColName).metadata()

        return dataset?.select(dataset.col("*"),
                functions.callUDF("lemmatizer", JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
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