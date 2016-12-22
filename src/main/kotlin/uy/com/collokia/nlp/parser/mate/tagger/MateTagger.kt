@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.tagger

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
import uy.com.collokia.nlp.parser.mate.lemmatizer.LematizerWrapper
import uy.com.collokia.nlp.parser.mate.lemmatizer.englishLemmatizerModelName
import uy.com.collokia.nlp.parser.mate.lemmatizer.spanishLemmatizerModelName
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable

const val englishTaggerModelName = "./data/mate/models/english/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model"
const val spanishTaggerModelName = "./data/mate/models/spanish/CoNLL2009-ST-Spanish-ALL.anna-3.3.postagger.model"

class MateTagger : Transformer, Serializable {

    val lemmatizerWrapper: LematizerWrapper
    val taggerWrapper: TaggerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val udfName = "tagger"
    val isEnglish: Boolean


    constructor(sparkSession: SparkSession, isEnglish: Boolean = true, inputColName: String = tokenizedContent) {

        this.isEnglish = isEnglish
        this.sparkSession = sparkSession

        val lemmatizerModel = if (isEnglish) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)

        val taggerModelName = if (isEnglish) englishTaggerModelName else spanishTaggerModelName
        taggerWrapper = TaggerWrapper(arrayOf("-model", taggerModelName))
        this.inputColName = inputColName
        this.outputColName = "taggedContent"


        val tagger = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
            val lemmatizer = lemmatizerWrapper.get()
            val posTagger = taggerWrapper.get()

            val sentencesJava = JavaConversions.asJavaCollection(sentences).filter { sentence -> sentence.size() < 100 }
            val results = arrayOfNulls<Array<Array<String>>>(sentencesJava.size)


            sentencesJava.forEachIndexed { sentenceNum, tokens ->

                val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                sentenceArray[0] = "<root>"

                (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                val lemmatized = SentenceData09()
                lemmatized.init(sentenceArray)
                lemmatizer.apply(lemmatized).plemmas.toList()
                val lemmas = lemmatized.plemmas

                var tagged = posTagger.tag(lemmatized)
                val posses = tagged.ppos

                val taggedValues = sentenceArray.mapIndexed { tokenIndex, token ->
                    arrayOf(token ?: "",lemmas[tokenIndex],posses[tokenIndex])
                }.toTypedArray()

                results[sentenceNum] = taggedValues
            }
            results
        })

        this.sparkSession.udf().register(udfName, tagger, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType))))
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
