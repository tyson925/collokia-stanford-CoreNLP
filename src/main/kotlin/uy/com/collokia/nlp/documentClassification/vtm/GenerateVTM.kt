package uy.com.collokia.nlp.documentClassification.vtm


//import org.apache.spark.mllib.linalg.SparseVector
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.ml.regression.LabeledPoint
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import scala.Tuple2
import uy.com.collokia.common.utils.nlp.*
import uy.com.collokia.nlp.transformer.OwnNGram


@Suppress("UNUSED_VARIABLE")

        //data class SimpleDocument(var category: String, var content: String, var title: String, var labels: String) : Serializable

fun extractFeaturesFromCorpus(textDataFrame: Dataset<*>,
                              categoryColName: String = "category",
                              contentColName: String = "content"): Dataset<Row> {

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

fun constructNgramsPipeline(stages: Array<PipelineStage>): Pipeline {
    val pipeline = Pipeline().setStages(stages)
    return pipeline
}

fun constructNgrams(stopwords: Set<String> = setOf(),
                    inputColName: String = "content",
                    toLowercase : Boolean = false,
                    minTokenLength : Int = 2): Array<PipelineStage> {

    val tokenizer = RegexTokenizer().setInputCol(inputColName).setOutputCol(inputColName + "_" + tokenizerOutputCol)
            .setMinTokenLength(minTokenLength)
            .setToLowercase(toLowercase)
            .setPattern("\\w+")
            .setGaps(false)

    val stopwordsApplied = if (stopwords.isEmpty()) {
        println("Load default english stopwords...")
        StopWordsRemover.loadDefaultStopWords("english")
    } else {
        println("Load stopwords...")
        stopwords.toTypedArray()
    }

    val remover = StopWordsRemover().setInputCol(tokenizer.outputCol).setOutputCol(inputColName + "_" + removeOutputCol)
            .setStopWords(stopwordsApplied)
            .setCaseSensitive(false)

    val ngram = OwnNGram().setInputCol(remover.outputCol).setOutputCol(inputColName + "_" + ngramOutputCol)

    return arrayOf(tokenizer, remover, ngram)
}

fun constructVTM(vocabSize: Int = CONTENT_VTM_VOC_SIZE, vtmInputCol: String): Array<PipelineStage> {

    val prefix = if (vtmInputCol.contains("_")){
        vtmInputCol.split("_").first()
    } else {
        vtmInputCol
    }

    val cvModel = CountVectorizer().setInputCol(vtmInputCol)
            .setVocabSize(vocabSize)
            .setMinDF(3.0)
            .setOutputCol(prefix + "_" + cvModelOutputCol)

    //it is useless
    //val idf = IDF().setInputCol(cvModel.outputCol).setOutputCol("idfFeatures").setMinDocFreq(3)

    val normalizer = Normalizer().setInputCol(cvModel.outputCol).setOutputCol(prefix + "_" + normalizerOutputCol).setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(normalizer.outputCol)
            .setOutputCol(prefix + "_" + contentOutputCol)
            .setWithStd(true)
            .setWithMean(false)

    return arrayOf(cvModel,normalizer,scaler)
}

fun constructVTMPipeline(stopwords: Array<String>,
                         vocabSize: Int,
                         inputColName: String = "content",
                         isTest: Boolean = false): Pipeline {

    val indexer = StringIndexer().setInputCol("category").setOutputCol(labelIndexCol)

    val ngramPipline = constructNgrams(stopwords.toSet(), inputColName)


    val tokenizer = ngramPipline[0] as Tokenizer
    val remover = ngramPipline[1] as StopWordsRemover
    val ngram = ngramPipline[2] as OwnNGram

    val vtm = constructVTM(vocabSize,ngram.outputCol)

    val cvModel = vtm[0] as CountVectorizer

    val normalizer = vtm[1] as Normalizer

    val scaler = vtm[2] as StandardScaler

    val pipeline = if (isTest) {
        Pipeline().setStages(arrayOf(tokenizer, remover, ngram, cvModel, normalizer, scaler))
    } else {
        Pipeline().setStages(arrayOf(indexer, tokenizer, remover, ngram, cvModel, normalizer, scaler))
    }

    return pipeline
}

/*fun constructTitleVtmDataPipeline(stopwords: Array<String>, vocabSize: Int): Pipeline {

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
}*/

fun constructTagVtmDataPipeline(vocabSize: Int, inputColName: String = "labels"): Pipeline {
    val tagTokenizer = RegexTokenizer().setInputCol(inputColName).setOutputCol(inputColName + "_" + tokenizerOutputCol)
            .setMinTokenLength(2)
            .setToLowercase(true)
            .setPattern("\\w+")
            .setGaps(false)

    val tagCVModel = CountVectorizer().setInputCol(tagTokenizer.outputCol)
            .setOutputCol(inputColName + "_" + cvModelOutputCol)
            .setVocabSize(vocabSize)
            .setMinDF(1.0)

    val tagNormalizer = Normalizer().setInputCol(tagCVModel.outputCol)
            .setOutputCol(inputColName + "_" + normalizerOutputCol)
            .setP(1.0)

    val scaler = StandardScaler()
            .setInputCol(tagCVModel.outputCol)
            .setOutputCol(inputColName + "_" + contentOutputCol)
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


fun loadStopwords(jsc: JavaSparkContext, stopwordsFileName: String = "./data/stopwords.txt"): Broadcast<Array<String>> {

    val stopwords = jsc.broadcast(jsc.textFile(stopwordsFileName).collect().toTypedArray())
    return stopwords
}