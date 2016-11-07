package uy.com.collokia.nlp.parser.mate

import is2.data.SentenceData09
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent


class MateTagger : Transformer {

    val taggerWrapper: TaggerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession

    constructor(sparkSession: SparkSession,
                taggerModelName : String = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model"){

        this.sparkSession = sparkSession
        taggerWrapper = TaggerWrapper(arrayOf("-model",taggerModelName))
        inputColName = tokenizedContent
        outputColName = ""
    }

    fun setInputColName(inputColName: String): MateTagger {
        this.inputColName = inputColName
        return this
    }

    override fun uid(): String {
        return "lemmatizer111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateTagger(sparkSession)
    }

    override fun transform(p0: Dataset<*>?): Dataset<Row> {
        val tagger = org.apache.spark.sql.api.java.UDF1({ tokens: scala.collection.mutable.WrappedArray<String> ->

            val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

            val lemmatized = SentenceData09()

            lemmatized.lemmas = sentenceArray

        })
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transformSchema(schema : StructType?): StructType {
        val inputType = schema?.apply(schema.fieldIndex(inputColName))
        val inputTypeMetaData = inputType?.metadata()
        val refType = DataTypes.createArrayType(DataTypes.StringType).javaClass

        if (inputTypeMetaData is DataTypes) {
            println("Input type must be StringType but got $inputTypeMetaData.")
        }
        return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.StringType), inputType?.nullable() ?: false)
    }

}
