package uy.com.collokia.nlp.documentClassification


import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.nlp.*
import uy.com.collokia.common.utils.rdd.readDzoneDataFromJson
import uy.com.collokia.nlp.documentClassification.vtm.constructTagVtmDataPipeline
import uy.com.collokia.nlp.documentClassification.vtm.constructTitleVtmDataPipeline
import uy.com.collokia.nlp.documentClassification.vtm.constructVTMPipeline
import uy.com.collokia.nlp.parser.mate.lemmatizedContentCol
import java.text.DecimalFormat

const val OVR_MODEL = "./data/model/ovrDectisonTree"
const val LABELS = "./data/model/labelIndexer_2"
const val REUTERS_DATA = "./data/reuters/json/reuters.json"
const val VTM_PIPELINE = "./data/model/vtmPipeLine"


const val corpusFileName = "./data/classification/dzone/dzone.parquet"
val formatter = DecimalFormat("#0.00")

fun generateDzoneVTM(jsc: JavaSparkContext, corpus: Dataset<Row>, tagColName: String = tagInputColName): Dataset<Row> {
    val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

    println("CORPUS SIZE:\t${corpus.count()}")

    val dataset = generateVTM(corpus, stopwords, isRunLocal = true, tagColName = tagColName)
    return dataset
}

fun generateDzoneVTM(jsc: JavaSparkContext, sparkSession: SparkSession, tagColName: String = tagInputColName): Dataset<Row> {

    val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

    val corpus = readDzoneDataFromJson(sparkSession, jsc)

    val dataset = generateVTM(corpus.toDF(), stopwords, isRunLocal = true, tagColName = tagColName)
    return dataset
}

fun generateVTM(corpus: Dataset<Row>,
                stopwords: Broadcast<Array<String>>,
                isRunLocal: Boolean,
                tagColName: String = tagInputColName): Dataset<Row> {

    val vtmDataPipeline = constructVTMPipeline(stopwords.value, CONTENT_VTM_VOC_SIZE, lemmatizedContentCol)

    println(corpus.count())

    val vtmPipelineModel = vtmDataPipeline.fit(corpus)

    val cvModel = vtmPipelineModel.stages()[4] as CountVectorizerModel
    println("cv model vocabulary: " + cvModel.vocabulary().toList())

    val indexer = vtmPipelineModel.stages()[0] as StringIndexerModel

    if (isRunLocal) {
        if (deleteIfExists(LABELS)) {
            println("save labels...")
            indexer.save(LABELS)
        }
    }

    val parsedCorpus = vtmPipelineModel.transform(corpus).drop(
            lemmatizedContentCol,
            tokenizerOutputCol,
            removeOutputCol,
            ngramOutputCol,
            cvModelOutputCol)

    val vtmTitlePipeline = constructTitleVtmDataPipeline(stopwords.value, TITLE_VTM_VOC_SIZE)

    val vtmTitlePipelineModel = vtmTitlePipeline.fit(parsedCorpus)

    val parsedCorpusTitle = vtmTitlePipelineModel.transform(parsedCorpus).drop(
            titleTokenizerOutputCol,
            titleRemoverOutputCol,
            titleNgramsOutputCol,
            titleCvModelOutputCol)

    //parsedCorpusTitle.show(10, false)

    val vtmTagPipeline = constructTagVtmDataPipeline(TAG_VTM_VOC_SIZE, tagColName)

    val vtmTagPipelineModel = vtmTagPipeline.fit(parsedCorpusTitle)

    val fullParsedCorpus = vtmTagPipelineModel.transform(parsedCorpusTitle).drop(tagTokenizerOutputCol, tagCvModelOutputCol)

    val contentScaler = vtmPipelineModel.stages().last() as StandardScalerModel

    val titleNormalizer = vtmTitlePipelineModel.stages().last() as StandardScalerModel

    val tagNormalizer = vtmTagPipelineModel.stages().last() as StandardScalerModel

    val assembler = VectorAssembler().setInputCols(arrayOf(contentScaler.outputCol, titleNormalizer.outputCol, tagNormalizer.outputCol))
            .setOutputCol(featureCol)

    val dataset = assembler.transform(fullParsedCorpus)

    if (isRunLocal) {
        if (deleteIfExists(VTM_PIPELINE)) {
            vtmPipelineModel.save(VTM_PIPELINE)
        }
//        dataset.write().save(corpusFileName)
    } else {

    }

    return dataset

}


fun evaluateModel10Fold(pipeline: Pipeline, corpus: Dataset<Row>) {
    val nFolds = 10
    val paramGrid = ParamGridBuilder().build() // No parameter search

    val evaluator = MulticlassClassificationEvaluator()
            .setLabelCol(labelIndexCol)
            .setPredictionCol(predictionCol)
            // "f1", "precision", "recall", "weightedPrecision", "weightedRecall"
            .setMetricName("f1")

    val crossValidator = CrossValidator()
            // ml.Pipeline with ml.classification.RandomForestClassifier
            .setEstimator(pipeline)
            // ml.evaluation.MulticlassClassificationEvaluator
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(nFolds)

    val crossValidatorModel = crossValidator.fit(corpus) // corpus: DataFrame

    val bestModel = crossValidatorModel.bestModel()

    val avgMetrics = crossValidatorModel.avgMetrics()

    val paramsToScore = crossValidatorModel.estimatorParamMaps.mapIndexed { i, paramMap ->
        Tuple2(paramMap, avgMetrics[i])
    }.sortedByDescending { stat -> stat._2 }

    println(paramsToScore.joinToString("\n"))
}


