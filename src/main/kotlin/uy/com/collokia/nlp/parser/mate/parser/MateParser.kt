package uy.com.collokia.nlp.parser.mate.parser

import com.collokia.resources.MATE_PARSER_RESOURCES_PATH_ES
import com.collokia.resources.MATE_STANFORD_RESOURCES_PATH_EN
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
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.mate.lemmatizer.LematizerWrapper
import uy.com.collokia.nlp.parser.mate.lemmatizer.englishLemmatizerModelName
import uy.com.collokia.nlp.parser.mate.lemmatizer.spanishLemmatizerModelName
import uy.com.collokia.nlp.parser.mate.tagger.TaggerWrapper
import uy.com.collokia.nlp.parser.mate.tagger.englishTaggerModelName
import uy.com.collokia.nlp.parser.mate.tagger.spanishTaggerModelName
import uy.com.collokia.nlp.parser.openNLP.tokenizedContent
import java.io.Serializable
import java.util.*

//const val englishParserModelName = "./data/mate/models/english/CoNLL2009-ST-English-ALL.anna-3.3.parser.model"
//"mate/models/english/stanford.model"
val englishParserModelName: String by lazy {
    ResourceUtil.getResourceAsFile(MATE_STANFORD_RESOURCES_PATH_EN).absolutePath
}
//"mate/models/spanish/CoNLL2009-ST-Spanish-ALL.anna-3.3.parser.model"
val spanishParsedModelName: String by lazy {
    ResourceUtil.getResourceAsFile(MATE_PARSER_RESOURCES_PATH_ES).absolutePath
}

data class ParsedToken(var token: String, var lemma: String, var posTag: String, var parse: String, var head: Int) : Serializable

data class ParsedSentence(var parsedSentence: List<ParsedToken>) : Serializable

data class ParsedContent(var parsedContent: List<ParsedSentence>) : Serializable


class MateParser : Transformer {

    val lemmatizerWrapper: LematizerWrapper
    val taggerWrapper: TaggerWrapper
    val parserWrapper: ParserWrapper
    var inputColName: String
    var outputColName: String
    val sparkSession: SparkSession
    val udfName = "parser"
    val language: LANGUAGE


    constructor(sparkSession: SparkSession, language: LANGUAGE = LANGUAGE.ENGLISH, inputColName: String = tokenizedContent) {

        this.sparkSession = sparkSession
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        lemmatizerWrapper = LematizerWrapper(arrayOf("-model", lemmatizerModel))

        val taggerModelName = if (language == LANGUAGE.ENGLISH) englishTaggerModelName else spanishTaggerModelName
        taggerWrapper = TaggerWrapper(arrayOf("-model", taggerModelName))

        val parserModelName = if (language == LANGUAGE.ENGLISH) englishParserModelName else spanishParsedModelName
        parserWrapper = ParserWrapper(arrayOf("-model", parserModelName))

        this.inputColName = inputColName
        this.outputColName = "taggedContent"
        this.language = language

        val parserUDF = UDF1({ sentences: WrappedArray<WrappedArray<String>> ->
            val lemmatizer = lemmatizerWrapper.get()
            val posTagger = taggerWrapper.get()
            val parser = parserWrapper.get()

            val sentencesJava = JavaConversions.asJavaCollection(sentences).filter { sentence -> sentence.size() < 80 }
            val results = ArrayList<Array<Array<String>>>(sentencesJava.size)

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

                val parsed = parser.apply(tagged)

                val parses = parsed.plabels
                val heads = parsed.pheads


                val taggedValues = sentenceArray.mapIndexed { tokenIndex, token ->
                    //val parserIndex = tokenIndex - 1
                    arrayOf(
                            token ?: "",
                            lemmas[tokenIndex] ?: "",
                            posses[tokenIndex] ?: "",
                            if (tokenIndex == 0 || tokenIndex > parses.size) "root" else parses[tokenIndex - 1] ?: "",
                            if (tokenIndex == 0 || tokenIndex > heads.size) "0" else heads[tokenIndex - 1].toString())
                }.toTypedArray()

                results.add(sentenceNum, taggedValues)
            }
            results
        })

        sparkSession.udf().register(udfName, parserUDF, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType))))

    }

    fun setInputColName(inputColName: String): MateParser {
        this.inputColName = inputColName
        return this
    }

    fun setOutputColName(outputColName: String): MateParser {
        this.outputColName = outputColName
        return this
    }

    override fun uid(): String {
        return "parser111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateParser(sparkSession, language, inputColName)
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
        return SchemaUtils.appendColumn(schema, outputColName, DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType))), inputType?.nullable() ?: false)
    }
}
