@file:Suppress("unused")

package uy.com.collokia.nlp.parser

import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import uy.com.collokia.common.data.dataClasses.stackoverflow.SoLitleModel.SOThreadExtractValues
import uy.com.collokia.nlp.parser.mate.lemmatizer.MateLemmatizer
import uy.com.collokia.nlp.parser.mate.parser.MateParser
import uy.com.collokia.nlp.parser.mate.tagger.MateTagger
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer


enum class PARSER_TYPE {
    TOKENIZER, LEMMATIZER, POSTAGGER, PARSER
}

enum class LANGUAGE {
    ENGLISH, SPANISH
}

fun lemmatizeContent(sparkSession: SparkSession,
                     dataset: Dataset<Row>,
                     inputColName: String = SOThreadExtractValues::content.name,
                     language: LANGUAGE = LANGUAGE.ENGLISH): Dataset<Row> {

    val tokenizer = OpenNlpTokenizer(sparkSession, isOutputRaw = false, language = language).setInputColName(inputColName)
    val lemmatizer = MateLemmatizer(sparkSession, isRawOutput = false, isRawInput = false, language = language).setInputColName(tokenizer.outputColName)
    val textAnalyzer = Pipeline().setStages(arrayOf(tokenizer, lemmatizer))

    val analyzer = textAnalyzer.fit(dataset)
    val analyzedData = analyzer.transform(dataset).drop(tokenizer.outputColName, inputColName)
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