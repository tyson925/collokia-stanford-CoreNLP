@file:Suppress("unused")

package uy.com.collokia.nlp.parser


import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.MapType
import org.apache.spark.sql.types.StructType
import uy.com.collokia.common.data.dataClasses.stackoverflow.SoLitleModel.SOThreadExtractValues
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizerRaw
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.mate.tagger.MateTagger
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import uy.com.collokia.nlp.transformer.ngram.NGramOnSentenceData
import java.io.Serializable

data class NLPToken(var index: Int, var token: String, var lemma: String?, var posTag: String?,
                    var indexInContent: Int, var parseTag: String?, var parseIndex: Int?) : Serializable, scala.Equals {

    override fun canEqual(p0: Any?): Boolean {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }
}

data class NLPSentence(var lemmatizedSentence: List<NLPToken>) : Serializable

data class NLPContent(var lemmatizedContent: List<NLPSentence>) : Serializable

enum class PARSER_TYPE {
    TOKENIZER, LEMMATIZER, POSTAGGER, PARSER
}

enum class LANGUAGE {
    ENGLISH, SPANISH
}

const val DEFAULT_NGRAM_SEPARATOR = "-"

fun lemmaType() : StructType {
    val token = DataTypes.createStructField("token", DataTypes.StringType, true)
    val lemma = DataTypes.createStructField("lemma", DataTypes.StringType, true)
    return DataTypes.createStructType(listOf(token, lemma))
}

fun nlpTokenType(): MapType {
    val index = DataTypes.createStructField("index", DataTypes.IntegerType, true)
    val token = DataTypes.createStructField("token", DataTypes.StringType, true)
    val lemma = DataTypes.createStructField("lemma", DataTypes.StringType, true)
    val posTag = DataTypes.createStructField("posTag", DataTypes.StringType, true)
    val indexInContent = DataTypes.createStructField("indexInContent", DataTypes.IntegerType, true)
    val parseTag = DataTypes.createStructField("parseTag", DataTypes.StringType, true)
    val parseIndex = DataTypes.createStructField("parseIndex", DataTypes.IntegerType, true)

    val map = DataTypes.createMapType(DataTypes.StringType, DataTypes.StringType)

    //return DataTypes.createStructType(listOf(index, token, lemma, posTag, indexInContent, parseTag, parseIndex))
    return map

}

fun tokenizeContent(sparkSession: SparkSession,
                    dataset: Dataset<Row>,
                    inputColName: String = SOThreadExtractValues::content.name,
                    language: LANGUAGE = LANGUAGE.ENGLISH,
                    isOutputRaw: Boolean = false): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = isOutputRaw, language = language).setInputColName(inputColName)
    return tokenizer.transform(dataset)
}

fun lemmatizeContent(sparkSession: SparkSession,
                     dataset: Dataset<Row>,
                     inputColName: String = SOThreadExtractValues::content.name,
                     language: LANGUAGE = LANGUAGE.ENGLISH,
                     isOutputRaw: Boolean = false): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = isOutputRaw, language = language).setInputColName(inputColName)
    val lemmatizer = if (isOutputRaw) {
        MateLemmatizerRaw(sparkSession, isRawOutput = isOutputRaw)
    } else {
        MateLemmatizer(sparkSession, language = language).setInputColName(tokenizer.outputColName)
    }
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, lemmatizer))

    val analyzer = textAnalyzer.fit(dataset)
    val analyzedData = analyzer.transform(dataset).drop(tokenizer.outputColName, inputColName)
    return analyzedData
}

fun ngramLemmatizeContent(sparkSession: SparkSession,
                          dataset: Dataset<Row>,
                          inputColName: String = "content",
                          language: LANGUAGE = LANGUAGE.ENGLISH): Dataset<Row> {
    val tokenizer = OpenNlpTokenizer(sparkSession, language = language, isOutputRaw = false, inputColName = inputColName)
    val lemmatizer = MateLemmatizer(sparkSession, language = language, inputColName = tokenizer.outputColName)
    val ngram = NGramOnSentenceData().setInputCol(lemmatizer.outputColName)
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, lemmatizer, ngram))
    val analyzer = textAnalyzer.fit(dataset)

    val analyzedData = analyzer.transform(dataset).drop(inputColName, tokenizer.outputColName, lemmatizer.outputColName)
    return analyzedData
}

fun postTagContent(sparkSession: SparkSession,
                   dataset: Dataset<Row>,
                   inputColName: String = SOThreadExtractValues::content.name,
                   language: LANGUAGE = LANGUAGE.ENGLISH
): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = false, language = language).setInputColName(inputColName)
    val tagger = MateTagger(sparkSession, language = language).setInputColName(tokenizer.outputColName)
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, tagger))

    val analyzer = textAnalyzer.fit(dataset)
    val analyzedData = analyzer.transform(dataset).drop(tokenizer.outputColName, inputColName)
    return analyzedData
}

fun parseContent(sparkSession: SparkSession,
                 dataset: Dataset<Row>,
                 inputColName: String = SOThreadExtractValues::content.name,
                 language: LANGUAGE = LANGUAGE.ENGLISH
): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = false, language = language).setInputColName(inputColName)
    val parser = MateParser(sparkSession, language = language).setInputColName(tokenizer.outputColName)
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, parser))

    val analyzer = textAnalyzer.fit(dataset)
    val analyzedData = analyzer.transform(dataset).drop(tokenizer.outputColName, inputColName)
    return analyzedData
}