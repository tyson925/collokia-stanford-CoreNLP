@file:Suppress("unused")

package uy.com.collokia.nlp.parser


import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.MapType
import scala.collection.JavaConversions
import scala.collection.mutable.WrappedArray
import uy.com.collokia.common.data.dataClasses.stackoverflow.SoLitleModel.SOThreadExtractValues
import uy.com.collokia.nlp.parser.mate.lemmatizer.*
import uy.com.collokia.nlp.parser.mate.parser.ENGLISH_PARSER_MODEL_NAME
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.mate.parser.PORTUGUESE_PARSER_MODEL_NAME
import uy.com.collokia.nlp.parser.mate.parser.SPANISH_PARSER_MODEL_NAME
import uy.com.collokia.nlp.parser.mate.tagger.ENGLISH_TAGGER_MODEL_NAME
import uy.com.collokia.nlp.parser.mate.tagger.MateTagger
import uy.com.collokia.nlp.parser.mate.tagger.PORTUGUESE_TAGGER_MODEL_NAME
import uy.com.collokia.nlp.parser.mate.tagger.SPANISH_TAGGER_MODEL_NAME
import uy.com.collokia.nlp.parser.openNLP.*
import uy.com.collokia.nlp.transformer.ngram.NGramOnSentenceData
import java.io.Serializable

interface NLPToken : Serializable {
    var index: String
    var token: String
    var lemma: String
    var indexInContent: String
}

data class LemmaToken(override var index: String,
                      override var token: String,
                      override var lemma: String,
                      override var indexInContent: String) : Serializable,NLPToken

data class PosToken(override var index: String,
                    override var token: String,
                    override var lemma: String,
                    override var indexInContent: String,
                    var posTag: String) :Serializable, NLPToken

//data class PosToken(var index: Int, var token: String, var lemma: String, var indexInContent: Int, var posTag: String) :Serializable

data class ParseToken(override var index: String,
                      override var token: String,
                      override var lemma: String,
                      override var indexInContent: String,
                      var posTag: String,
                      var parseTag: String,
                      var parseIndex: String) :Serializable, NLPToken

data class NLPSentence(var Sentence: List<NLPToken>) : Serializable

data class NLPContent(var Content: List<NLPSentence>) : Serializable

enum class PARSER_TYPE {
    TOKENIZER, LEMMATIZER, POSTAGGER, PARSER
}

enum class LANGUAGE {
    ENGLISH, SPANISH, PORTUGUESE
}

const val DEFAULT_NGRAM_SEPARATOR = "-"


fun nlpTokenType(): MapType {

    return DataTypes.createMapType(DataTypes.StringType, DataTypes.StringType)
}

fun toNLPContentRDD(dataset: Dataset<Row>,parserType: PARSER_TYPE) : JavaRDD<NLPContent> {
    return dataset.toJavaRDD().map { row ->
        val parsedSentences = row.getList<WrappedArray<scala.collection.immutable.Map<String, String>>>(0)
        NLPContent(parsedSentences.map { sentence ->
            NLPSentence(JavaConversions.asJavaCollection(sentence).map { map ->

                getToken(map,parserType)
            })
        })
    }
}

fun getSentenceSplitterModelName(language: LANGUAGE) : String {
    return if (language == LANGUAGE.ENGLISH) {
        EN_SENTENCE_MODEL_NAME
    } else if (language == LANGUAGE.SPANISH) {
        ES_SENTENCE_MODEL_NAME
    } else if (language == LANGUAGE.PORTUGUESE) {
        PT_SENTENCE_MODEL_NAME
    } else {
        ""
    }
}
fun getTokenizerModelName(language: LANGUAGE) : String {
    return if (language == LANGUAGE.ENGLISH) {
        EN_TOKEN_MODEL_NAME
    } else if (language == LANGUAGE.SPANISH) {
        ES_TOKEN_MODEL_NAME
    } else if (language == LANGUAGE.PORTUGUESE) {
        PT_TOKEN_MODEL_NAME
    } else {
        ""
    }
}

fun getLemmatizerModelName(language: LANGUAGE) : String{
    return if (language == LANGUAGE.ENGLISH) {
        ENGLISH_LEMMATIZER_MODEL_NAME
    } else if (language == LANGUAGE.SPANISH) {
        SPANISH_LEMMATIZER_MODEL_NAME
    } else if (language == LANGUAGE.PORTUGUESE) {
        PORTUGUESE_LEMMATIZER_MODEL_NAME
    } else {
        ENGLISH_LEMMATIZER_MODEL_NAME
    }
}

fun getTaggerModelName(language: LANGUAGE) : String {
    return if (language == LANGUAGE.ENGLISH){
        ENGLISH_TAGGER_MODEL_NAME
    }else if (language == LANGUAGE.SPANISH) {
        SPANISH_TAGGER_MODEL_NAME
    } else if (language == LANGUAGE.PORTUGUESE) {
        PORTUGUESE_TAGGER_MODEL_NAME
    } else {
        ENGLISH_TAGGER_MODEL_NAME
    }
}

fun getParserModelName(language: LANGUAGE) : String {
    return if (language == LANGUAGE.ENGLISH){
        ENGLISH_PARSER_MODEL_NAME
    }else if (language == LANGUAGE.SPANISH) {
        SPANISH_PARSER_MODEL_NAME
    } else if (language == LANGUAGE.PORTUGUESE) {
        PORTUGUESE_PARSER_MODEL_NAME
    } else {
        ENGLISH_PARSER_MODEL_NAME
    }
}

fun getToken(map: scala.collection.immutable.Map<String, String>, parserType: PARSER_TYPE): NLPToken {

    return if (parserType == PARSER_TYPE.LEMMATIZER) {
        LemmaToken(index = map[NLPToken::index.name].get(),
                token = map[NLPToken::token.name].get(),
                lemma = map[NLPToken::lemma.name].get(),
                indexInContent = map[NLPToken::indexInContent.name].get())

    } else if (parserType == PARSER_TYPE.POSTAGGER) {
        PosToken(index = map[PosToken::index.name].get(),
                token = map[PosToken::token.name].get(),
                lemma = map[PosToken::lemma.name].get(),
                indexInContent = map[PosToken::indexInContent.name].get(),
                posTag = map[PosToken::posTag.name].get())
    } else {
        ParseToken(index = map[ParseToken::index.name].get(),
                token = map[ParseToken::token.name].get(),
                lemma = map[ParseToken::lemma.name].get(),
                indexInContent = map[ParseToken::indexInContent.name].get(),
                posTag = map[ParseToken::posTag.name].get(),
                parseTag = map[ParseToken::parseTag.name].get(),
                parseIndex = map[ParseToken::parseIndex.name].get())
    }
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
                     isOutputRaw: Boolean = false,
                     isDrop : Boolean = true): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = isOutputRaw, language = language).setInputColName(inputColName)
    val lemmatizer = if (isOutputRaw) {
        MateLemmatizerRaw(sparkSession, isRawOutput = isOutputRaw, inputColName = tokenizer.outputColName, outputColName = inputColName)
    } else {
        MateLemmatizer(sparkSession, language = language, inputColName = tokenizer.outputColName, outputColName = inputColName)
    }
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, lemmatizer))

    val analyzer = textAnalyzer.fit(dataset)
    val analyzedData = if (isDrop) {
        analyzer.transform(dataset).drop(tokenizer.outputColName, inputColName)
    } else {
        analyzer.transform(dataset).drop(tokenizer.outputColName)
    }
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