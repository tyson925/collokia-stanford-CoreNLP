@file:Suppress("unused", "UNUSED_VARIABLE")

package uy.com.collokia.nlp.documentClassification

import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.machineLearning.FEATURE_COL_NAME
import uy.com.collokia.common.utils.machineLearning.LABEL_COL_NAME
import uy.com.collokia.common.utils.machineLearning.PREDICTION_COL_NAME
import uy.com.collokia.common.utils.nlp.*
import uy.com.collokia.common.utils.rdd.readDzoneDataFromJson
import uy.com.collokia.nlp.documentClassification.vtm.*
import uy.com.collokia.nlp.parser.mate.lemmatizer.lemmatizedContentCol
import uy.com.collokia.nlp.transformer.ngram.OwnNGram

const val OVR_MODEL = "./data/model/ovrDectisonTree"
const val LABELS = "./data/model/labelIndexer_2"
const val REUTERS_DATA = "./data/reuters/json/reuters.json"
const val VTM_PIPELINE_MODEL_NAME = "./data/model/vtmPipeLine"
const val TITLE_PIPELINE_MODEL_NAME = "./data/model/titlePipeLine"
const val TAGS_PIPELINE_MODEL_NAME = "./data/model/tagsPipeLine"

//const val corpusFileName = "./data/classification/dzone/dzone.parquet"
//val formatter = DecimalFormat("#0.00")

fun generateDzoneVTM(jsc: JavaSparkContext, corpus: Dataset<Row>, tagColName: String = tagInputColName, isTest: Boolean = false)
        : Dataset<Row> {

    val stopwords = loadStopwords(jsc)

    println("CORPUS SIZE:\t${corpus.count()}")

    val dataset = generateVTM(corpus, stopwords, isRunLocal = true, tagColName = tagColName, isTest = isTest)
    return dataset
}

fun generateDzoneVTM(jsc: JavaSparkContext, sparkSession: SparkSession, tagColName: String = tagInputColName): Dataset<Row> {

    val stopwords = loadStopwords(jsc)

    val corpus = readDzoneDataFromJson(sparkSession, jsc)

    val dataset = generateVTM(corpus.toDF(), stopwords, isRunLocal = true, tagColName = tagColName)
    return dataset
}


fun generateVTM(corpus: Dataset<Row>,
                stopwords: Broadcast<Array<String>>,
                isRunLocal: Boolean,
                tagColName: String = tagInputColName,
                isTest: Boolean = false): Dataset<Row> {

    val ngramPipe = constructNgramsPipeline(constructNgrams(stopwords = stopwords.value.toSet(),
            inputColName = lemmatizedContentCol,
            toLowercase = true))

    val parsedCorpus = ngramPipe.fit(corpus).transform(corpus).drop(lemmatizedContentCol,
            lemmatizedContentCol + "_" + tokenizerOutputCol,
            lemmatizedContentCol + "_" + removeOutputCol)

    val vtmModel = loadPipelineModel(isTest = isTest,
            isRunLocal = isRunLocal,
            corpus = parsedCorpus,
            modelName = VTM_PIPELINE_MODEL_NAME,
            vtmSize = CONTENT_VTM_VOC_SIZE,
            inputColName = (ngramPipe.stages.last() as OwnNGram).outputCol)


    val vtm = vtmModel.transform(parsedCorpus).drop(lemmatizedContentCol + "_" + ngramOutputCol,
            lemmatizedContentCol + "_" + cvModelOutputCol)

    val indexer = StringIndexer().setInputCol("category").setOutputCol(LABEL_COL_NAME)

    val indexerModel = indexer.fit(vtm)

    val indexedVTM = if (isTest) {
        vtm
    } else {
        indexerModel.transform(vtm)
    }

    if (isRunLocal) {
        if (deleteIfExists(LABELS)) {
            println("save labels...")
            indexerModel.save(LABELS)
        }
    }

    val titleNgramPipe = constructNgramsPipeline(constructNgrams(stopwords = stopwords.value.toSet(),
            inputColName = "title",
            toLowercase = false))

    val parsedCorpusTitle = titleNgramPipe.fit(indexedVTM).transform(indexedVTM).drop(
            "title_" + tokenizerOutputCol,
            "title_" + removeOutputCol)

    println("title output colname: " + (titleNgramPipe.stages.last() as OwnNGram).outputCol)

    val vtmTitlePipelineModel = loadPipelineModel(isTest = isTest,
            isRunLocal = isRunLocal,
            corpus = parsedCorpusTitle,
            modelName = TITLE_PIPELINE_MODEL_NAME,
            vtmSize = TITLE_VTM_VOC_SIZE,
            inputColName = (titleNgramPipe.stages.last() as OwnNGram).outputCol)


    val vtmTitleCorpus = vtmTitlePipelineModel.transform(parsedCorpusTitle).drop("title_" + ngramOutputCol,
            "title_" + cvModelOutputCol)



    val vtmTagPipelineModel = if (isTest) {
        PipelineModel.load(TAGS_PIPELINE_MODEL_NAME)
    } else {
        println(vtmTitleCorpus.schema())
        val vtmTagPipeline = constructTagVtmDataPipeline(TAG_VTM_VOC_SIZE, tagColName)
        val model = vtmTagPipeline.fit(vtmTitleCorpus)
        savePipelineModel(model, TAGS_PIPELINE_MODEL_NAME)
        model
    }

    val fullParsedCorpus = vtmTagPipelineModel.transform(vtmTitleCorpus).drop(tagColName + "_" + tokenizerOutputCol,
            tagColName + "_" + cvModelOutputCol)

    val contentScaler = vtmModel.stages().last() as StandardScalerModel

    val titleNormalizer = vtmTitlePipelineModel.stages().last() as StandardScalerModel

    val tagNormalizer = vtmTagPipelineModel.stages().last() as StandardScalerModel

    val assembler = VectorAssembler().setInputCols(arrayOf(contentScaler.outputCol,
            titleNormalizer.outputCol,
            tagNormalizer.outputCol))
            .setOutputCol(FEATURE_COL_NAME)

    val dataset = assembler.transform(fullParsedCorpus).drop(contentScaler.outputCol, titleNormalizer.outputCol, tagNormalizer.outputCol)

    return dataset
}

fun savePipelineModel(pipelineModel: PipelineModel, name: String) {
    if (deleteIfExists(name)) {
        pipelineModel.save(name)
    }
}

fun loadPipelineModel(isTest: Boolean = false,
                      isRunLocal: Boolean,
                      corpus: Dataset<Row>,
                      modelName: String,
                      vtmSize: Int,
                      inputColName: String): PipelineModel {


    return if (isTest) {
        PipelineModel.load(modelName)
    } else {
        val model = Pipeline().setStages(constructVTM(vtmSize, inputColName)).fit(corpus)
        if (isRunLocal) {
            savePipelineModel(model, modelName)
        }
        model
    }
}


fun evaluateModel10Fold(pipeline: Pipeline, corpus: Dataset<Row>) {
    val nFolds = 10
    val paramGrid = ParamGridBuilder().build() // No parameter search

    val evaluator = MulticlassClassificationEvaluator()
            .setLabelCol(LABEL_COL_NAME)
            .setPredictionCol(PREDICTION_COL_NAME)
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

    val prediction = crossValidatorModel.transform(corpus)

    val bestModel = crossValidatorModel.bestModel()

    bestModel

    val avgMetrics = crossValidatorModel.avgMetrics()

    val paramsToScore = crossValidatorModel.estimatorParamMaps.mapIndexed { i, paramMap ->
        Tuple2(paramMap, avgMetrics[i])
    }.sortedByDescending { stat -> stat._2 }

    println(paramsToScore.joinToString("\n"))
}


