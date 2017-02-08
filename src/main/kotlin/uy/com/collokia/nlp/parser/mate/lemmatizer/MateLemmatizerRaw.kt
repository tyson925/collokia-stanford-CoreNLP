package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.data.SentenceData09
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
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.openNLP.TOKENIZED_CONTENT_COL_NAME
import java.io.Serializable


class MateLemmatizerRaw : Transformer, Serializable {

    val lemmatizerWrapper: LematizerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val isRawOutput: Boolean
    var language: LANGUAGE
    val udfName = "lemmatizer"

    constructor(sparkSession: SparkSession,
                isRawOutput: Boolean,
                language: LANGUAGE = LANGUAGE.ENGLISH,
                inputColName: String = TOKENIZED_CONTENT_COL_NAME,
                outputColName: String = LEMMATIZED_CONTENT_COL_NAME) {

        this.sparkSession = sparkSession
        this.isRawOutput = isRawOutput
        this.language = language
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) ENGLISH_LEMMATIZER_MODEL_NAME else SPANISH_LEMMATIZER_MODEL_NAME
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        this.inputColName = inputColName
        this.outputColName = outputColName

        val rawLemmatizer = UDF1({ tokens: WrappedArray<String> ->
            val lemmatizer = lemmatizerWrapper.get()
            val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

            val lemmatized = SentenceData09()
            lemmatized.init(sentenceArray)

            if (this.isRawOutput) {
                lemmatizer.apply(lemmatized).plemmas.joinToString(" ")
            } else {
                lemmatizer.apply(lemmatized).plemmas.toList()
            }
        })

        if (isRawOutput) {
            sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.StringType)
        } else {
            sparkSession.udf().register(udfName, rawLemmatizer, DataTypes.createArrayType(DataTypes.StringType))
        }
    }

    fun setInputColName(inputColName: String): MateLemmatizerRaw {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): MateLemmatizerRaw {
        this.outputColName = outputColName
        return this
    }

    override fun uid(): String {
        return "lemmatizer111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateLemmatizerRaw(sparkSession, isRawOutput, language, inputColName, outputColName)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row> {

        //val outputDataType = transformSchema(dataset?.schema()).apply(outputColName).metadata()

        return dataset?.let {
            dataset.select(dataset.col("*"),
                    functions.callUDF(udfName, JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
        } ?: sparkSession.emptyDataFrame()
    }

    override fun transformSchema(schema: StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        //val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes) {
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        val nullable = inputType?.nullable() ?: false
        val output = if (isRawOutput) {
            DataTypes.StringType
        } else {
            DataTypes.createArrayType(DataTypes.StringType)
        }
        return SchemaUtils.appendColumn(schema, DataTypes.createStructField(outputColName,output,nullable))
    }
}