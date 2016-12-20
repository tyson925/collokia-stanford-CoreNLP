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
import java.io.Serializable

private const val englishTaggerModelName = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model"
private const val spanishTaggerModelName = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model"

class MateTagger : Transformer, Serializable {

    val taggerWrapper: TaggerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val udfName = "tagger"
    val isEnglish: Boolean


    constructor(sparkSession: SparkSession, isEnglish: Boolean = true, inputColName: String = tokenizedContent) {

        this.isEnglish = isEnglish

        this.sparkSession = sparkSession
        val modelName = if (isEnglish) englishTaggerModelName else spanishTaggerModelName
        taggerWrapper = TaggerWrapper(arrayOf("-model", modelName))
        this.inputColName = inputColName
        this.outputColName = "taggedContent"

        val tagger = org.apache.spark.sql.api.java.UDF1({ tokens: scala.collection.mutable.WrappedArray<String> ->

            val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

            val lemmatized = SentenceData09()

            lemmatized.lemmas = sentenceArray
        })

        this.sparkSession.udf().register(udfName, tagger, DataTypes.createArrayType(DataTypes.StringType))
    }



    fun setInputColName(inputColName: String): MateTagger {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): MateTagger {
        this.outputColName = outputColName
        return this
    }

    override fun uid(): String {
        return "mateTagger1111111111111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateTagger(sparkSession, isEnglish)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        return dataset?.select(dataset.col("*"),
                functions.callUDF(udfName, JavaConversions.asScalaBuffer(listOf(dataset.col(inputColName)))).`as`(outputColName))
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
