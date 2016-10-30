package uy.com.collokia.nlp.documentClassification.vtm


//import org.apache.spark.mllib.linalg.SparseVector
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.ml.regression.LabeledPoint
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import scala.Tuple2
import uy.com.collokia.common.utils.nlp.*
import uy.com.collokia.nlp.transformer.OwnNGram
import java.io.Serializable


@Suppress("UNUSED_VARIABLE")

//data class SimpleDocument(var category: String, var content: String, var title: String, var labels: String) : Serializable

fun extractFeaturesFromCorpus(textDataFrame: Dataset<*>,
                              categoryColName : String = "category",
                              contentColName : String = "content"): Dataset<Row> {

    val indexer = StringIndexer().setInputCol(categoryColName).setOutputCol(labelIndexCol).fit(textDataFrame)
    println(indexer.labels().joinToString("\t"))

    val indexedTextDataFrame = indexer.transform(textDataFrame)

    val tokenizer = Tokenizer().setInputCol(contentColName).setOutputCol(tokenizerOutputCol)
    val wordsDataFrame = tokenizer.transform(indexedTextDataFrame)

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(featureCol)

    val filteredWordsDataFrame = remover.transform(wordsDataFrame)

    //val ngramTransformer = NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(4)
    val ngramTransformer = OwnNGram().setInputCol(remover.outputCol).setOutputCol(ngramOutputCol)

    //       val ngramsDataFrame = ngramTransformer.transform(filteredWordsDataFrame)
    val ngramsDataFrame = ngramTransformer.transform(wordsDataFrame)

    //return ngramsDataFrame
    return filteredWordsDataFrame
}

fun constructNgrams(stopwords: Set<String>,inputColName: String): Pipeline {
    val tokenizer = RegexTokenizer().setInputCol(inputColName).setOutputCol(tokenizerOutputCol)
            .setMinTokenLength(2)
            .setToLowercase(false)
            .setPattern("\\w+")
            .setGaps(false)

    val stopwordsApplied = if (stopwords.isEmpty()) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords.toTypedArray()
    }

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(removeOutputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(remover.outputCol).setOutputCol(ngramOutputCol)

    val pipeline = Pipeline().setStages(arrayOf(tokenizer,remover, ngram))

    return pipeline
}

fun constructVTMPipeline(stopwords: Array<String>, vocabSize: Int, inputColName: String = "content"): Pipeline {
    val indexer = StringIndexer().setInputCol("category").setOutputCol(labelIndexCol)

    val tokenizer = RegexTokenizer().setInputCol(inputColName).setOutputCol(tokenizerOutputCol)
            .setMinTokenLength(3)
            .setToLowercase(false)
            .setPattern("\\w+")
            .setGaps(false)

    val stopwordsApplied = if (stopwords.isEmpty()) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords
    }

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(removeOutputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(remover.outputCol).setOutputCol(ngramOutputCol)

    val cvModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setVocabSize(vocabSize)
            .setMinDF(3.0)
            .setOutputCol(cvModelOutputCol)

    //it is useless
    //val idf = IDF().setInputCol(cvModel.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

    val normalizer = Normalizer().setInputCol(cvModel.outputCol).setOutputCol(contentOutputCol).setP(1.0)
    val scaler = StandardScaler()
            .setInputCol(cvModel.outputCol)
            .setOutputCol(contentOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(indexer, tokenizer, remover, ngram, cvModel, scaler))

    return pipeline
}

fun constructTitleVtmDataPipeline(stopwords: Array<String>, vocabSize: Int): Pipeline {

    val stopwordsApplied = if (stopwords.isEmpty()) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords
    }

    val titleTokenizer = RegexTokenizer().setInputCol("title").setOutputCol(titleTokenizerOutputCol)
            .setMinTokenLength(3)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    val titleRemover = StopWordsRemover().setInputCol(titleTokenizer.outputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)
            .setOutputCol(titleRemoverOutputCol)

    val ngram = OwnNGram().setInputCol(titleRemover.outputCol).setOutputCol(titleNgramsOutputCol)

    //val concatWs = ConcatWSTransformer().setInputCols(arrayOf(titleRemover.outputCol, ngram.outputCol)).setOutputCol("title_bigrams")

    val titleCVModel = CountVectorizer().setInputCol(ngram.outputCol)
            .setOutputCol(titleCvModelOutputCol)
            .setVocabSize(vocabSize)
            .setMinDF(2.0)

    val titleNormalizer = Normalizer().setInputCol(titleCVModel.outputCol)
            .setOutputCol(titleOutputCol)
            .setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(titleCVModel.outputCol)
            .setOutputCol(titleOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(titleTokenizer, titleRemover, ngram, titleCVModel, scaler))
    return pipeline
}

fun constructTagVtmDataPipeline(vocabSize: Int, inputColName : String = "labels"): Pipeline {
    val tagTokenizer = RegexTokenizer().setInputCol(inputColName).setOutputCol(tagTokenizerOutputCol)
            .setMinTokenLength(2)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    //val ngram = NGram().setInputCol(tagTokenizer.setOutputCol).setOutputCol("tag_ngrams").setN(3)

    val tagCVModel = CountVectorizer().setInputCol(tagTokenizer.outputCol)
            .setOutputCol(tagCvModelOutputCol)
            .setVocabSize(vocabSize)
            .setMinDF(1.0)

    val tagNormalizer = Normalizer().setInputCol(tagCVModel.outputCol)
            .setOutputCol(tagOutputCol)
            .setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(tagCVModel.outputCol)
            .setOutputCol(tagOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    val pipeline = Pipeline().setStages(arrayOf(tagTokenizer, tagCVModel, scaler))
    return pipeline

}


fun convertDataFrameToLabeledPoints(data: Dataset<Row>): JavaRDD<LabeledPoint> {
    val converter = IndexToString()
            .setInputCol(labelIndexCol)
            .setOutputCol("originalCategory")
    val converted = converter.transform(data)


    val featureData = converted.select(featureCol, labelIndexCol, "originalCategory")

    val labeledDataPoints = featureData.toJavaRDD().map({ feature ->
        val features = feature.getAs<SparseVector>(0)
        val label = feature.getDouble(1)
//            println(label)
        LabeledPoint(label, org.apache.spark.mllib.linalg.SparseVector(features.size(), features.indices(), features.values()))
    })

    println("number of testData: " + labeledDataPoints.count())

    val labelStat = featureData.select("originalCategory").javaRDD().mapToPair { label ->
        Tuple2(label.getString(0), 1L)
    }.reduceByKey { a, b -> a + b }

    println(labelStat.collectAsMap())

    return labeledDataPoints
}

fun setTfIdfModel(corpus: Dataset<Row>): IDFModel {
    val idf = IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures").setMinDocFreq(3)

    val idfModel = idf.fit(corpus)

    return idfModel
}


