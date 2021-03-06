@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.tagger

import com.collokia.resources.MATE_POSTAGGER_RESOURCES_PATH_EN
import com.collokia.resources.MATE_POSTAGGER_RESOURCES_PATH_ES
import com.collokia.resources.MATE_POSTAGGER_RESOURCES_PATH_PT
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
import uy.com.collokia.common.utils.resources.ResourceUtil
import uy.com.collokia.nlp.parser.*
import uy.com.collokia.nlp.parser.mate.lemmatizer.LematizerWrapper
import uy.com.collokia.nlp.parser.openNLP.TOKENIZED_CONTENT_COL_NAME
import java.io.Serializable
import java.util.*

//"mate/models/english/CoNLL2009-ST-English-ALL.anna-3.3.postagger.model"
val ENGLISH_TAGGER_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(MATE_POSTAGGER_RESOURCES_PATH_EN).absolutePath
}
//"mate/models/spanish/CoNLL2009-ST-Spanish-ALL.anna-3.3.postagger.model"
val SPANISH_TAGGER_MODEL_NAME: String  by lazy {
    ResourceUtil.getResourceAsFile(MATE_POSTAGGER_RESOURCES_PATH_ES).absolutePath
}

//"data/mate/models/portuguese/tagger_pt.model"
val PORTUGUESE_TAGGER_MODEL_NAME: String by lazy {
    ResourceUtil.getResourceAsFile(MATE_POSTAGGER_RESOURCES_PATH_PT).absolutePath
}

const val TAGGER_OUTPUT_COL_NAME = "taggedContent"

data class TaggedToken(var token: String, var lemma: String, var posTag: String) : Serializable

data class TaggedSentence(var taggedSentence: List<TaggedToken>) : Serializable

data class TaggedContent(var taggedContent: List<TaggedSentence>) : Serializable


class MateTagger : Transformer, Serializable {

    val lemmatizerWrapper: LematizerWrapper
    val taggerWrapper: TaggerWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val udfName = "tagger"
    val language: LANGUAGE


    constructor(sparkSession: SparkSession,
                language: LANGUAGE = LANGUAGE.ENGLISH,
                inputColName: String = TOKENIZED_CONTENT_COL_NAME,
                outputColName: String = TAGGER_OUTPUT_COL_NAME,
                lemmatizerModelName: String = "",
                taggerModelName: String = "") {

        this.language = language
        this.sparkSession = sparkSession

        val lemmatizerOptions = arrayOf("-model", if (lemmatizerModelName.isNotEmpty()) lemmatizerModelName else getLemmatizerModelName(language))
        lemmatizerWrapper = LematizerWrapper(lemmatizerOptions)

        val taggerOptions = arrayOf("-model", if (taggerModelName.isNotEmpty()) taggerModelName else getTaggerModelName(language))
        taggerWrapper = TaggerWrapper(taggerOptions)
        this.inputColName = inputColName
        this.outputColName = outputColName


        val tagger = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
            val lemmatizer = lemmatizerWrapper.get()
            val posTagger = taggerWrapper.get()

            val sentencesJava = JavaConversions.asJavaCollection(sentences).filter { sentence -> sentence.size() < 100 }
            val results = ArrayList<Array<Map<String, String>>>(sentences.size())

            sentencesJava.forEachIndexed { sentenceNum, tokens ->

                val sentenceArray = arrayOfNulls<String>(tokens.size() + 1) // according to the "root"

                sentenceArray[0] = "<root>"

                (0..tokens.size() - 1).forEach { i -> sentenceArray[i + 1] = tokens.apply(i) }

                val lemmatized = SentenceData09()
                lemmatized.init(sentenceArray)
                lemmatizer.apply(lemmatized).plemmas.toList()
                val lemmas = lemmatized.plemmas

                val tagged = posTagger.tag(lemmatized)
                val posses = tagged.ppos

                val contentIndex = results.map { sentence -> sentence.size }.sum()

                val taggedValues = sentenceArray.mapIndexed { tokenIndex, token ->
                    val values = mapOf(
                            NLPToken::index.name to tokenIndex.toString(),
                            NLPToken::token.name to (token ?: ""),
                            NLPToken::lemma.name to lemmas[tokenIndex],
                            NLPToken::indexInContent.name to (contentIndex + tokenIndex).toString(),
                            PosToken::posTag.name to posses[tokenIndex]
                    )
                    values
                }.toTypedArray()

                results.add(sentenceNum, taggedValues)
            }
            results
        })

        this.sparkSession.udf().register(udfName, tagger, DataTypes.createArrayType(DataTypes.createArrayType(nlpTokenType())))
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
        return MateTagger(sparkSession, language)
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row> {

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

        val token = DataTypes.createStructField("token", DataTypes.StringType, nullable)
        val lemma = DataTypes.createStructField("lemma", DataTypes.StringType, nullable)
        val posTag = DataTypes.createStructField("posTag", DataTypes.StringType, nullable)
        val output = DataTypes.createStructField(outputColName, DataTypes.createArrayType(DataTypes.createStructType(listOf(token, lemma, posTag))), nullable)
        return SchemaUtils.appendColumn(schema, outputColName, output.dataType(), nullable)
    }

}
