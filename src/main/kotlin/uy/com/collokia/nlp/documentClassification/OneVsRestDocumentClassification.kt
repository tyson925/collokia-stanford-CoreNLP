package uy.com.collokia.nlp.documentClassification


import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.OneVsRestModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import scala.Tuple2
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.machineLearning.EvaluationMetrics
import uy.com.collokia.common.utils.machineLearning.printMatrix
import uy.com.collokia.common.utils.rdd.readDzoneDataFromJson
import uy.com.collokia.nlp.documentClassification.vtm.*
import java.text.DecimalFormat

val OVR_MODEL = "./data/model/ovrDectisonTree"
val LABELS = "./data/model/labelIndexer"
val REUTERS_DATA = "./data/reuters/json/reuters.json"
val VTM_PIPELINE = "./data/model/vtmPipeLine"


val corpusFileName = "./data/classification/dzone/dzone.parquet"
val formatter = DecimalFormat("#0.00")

fun generateDzoneVTM(jsc: JavaSparkContext, corpus: Dataset<Row>, tagInputColName: String = SimpleDocument::labels.name): Dataset<Row> {
    val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

    println("CORPUS SIZE:\t${corpus.count()}")

    val dataset = generateVTM(corpus, stopwords, isRunLocal = true, tagInputColName = tagInputColName)
    return dataset
}

fun generateDzoneVTM(jsc: JavaSparkContext, sparkSession: SparkSession, tagInputColName: String = SimpleDocument::labels.name): Dataset<Row> {

    val stopwords = jsc.broadcast(jsc.textFile("./data/stopwords.txt").collect().toTypedArray())

    val corpus = readDzoneDataFromJson(sparkSession, jsc)

    val dataset = generateVTM(corpus.toDF(), stopwords, isRunLocal = true, tagInputColName = tagInputColName)
    return dataset
}

fun generateVTM(corpus: Dataset<Row>,
                stopwords: Broadcast<Array<String>>,
                isRunLocal: Boolean,
                tagInputColName: String = SimpleDocument::labels.name): Dataset<Row> {


    val vtmDataPipeline = constructVTMPipeline(stopwords.value, CONTENT_VTM_VOC_SIZE)

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
            SimpleDocument::content.name,
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

    parsedCorpusTitle.show(10, false)

    val vtmTagPipeline = constructTagVtmDataPipeline(TAG_VTM_VOC_SIZE, tagInputColName)

    val vtmTagPipelineModel = vtmTagPipeline.fit(parsedCorpusTitle)

    val fullParsedCorpus = vtmTagPipelineModel.transform(parsedCorpusTitle).drop(tagTokenizerOutputCol, tagCvModelOutputCol)

    val contentScaler = vtmPipelineModel.stages().last() as StandardScalerModel

    val titleNormalizer = vtmTitlePipelineModel.stages().last() as StandardScalerModel

    val tagNormalizer = vtmTagPipelineModel.stages().last() as StandardScalerModel

    //VectorAssembler().
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

fun evaluateModelConfusionMTX(ovrModel: OneVsRestModel, test: Dataset<Row>) {
    val indexer = StringIndexerModel.load(LABELS)

    val metrics = evaluateModel(ovrModel, test, indexer)
    val confusionMatrix = metrics.confusionMatrix()

// compute the false positive rate per label
//        val predictionColSchema = predictions.schema().fields()[0]
//        val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get()

    val fprs = (0..indexer.labels().size - 1).map({ p ->
        val fMeasure = metrics.fMeasure(p.toDouble()) * 100
        val precision = metrics.precision(p.toDouble()) * 100
        val recall = metrics.recall(p.toDouble()) * 100

        Tuple2(indexer.labels()[p], EvaluationMetrics(indexer.labels()[p], fMeasure, precision, recall))
    })

    println(printMatrix(confusionMatrix, indexer.labels().toList()))
    println("overall results:")
    println("FMeasure:\t${formatter.format(metrics.weightedFMeasure() * 100)}\t" +
            "Precision:\t${formatter.format(metrics.weightedPrecision() * 100)}\t" +
            "Recall:\t${formatter.format(metrics.weightedRecall() * 100)}\t" +
            "TP:\t${formatter.format(metrics.weightedTruePositiveRate() * 100)}\n" +
            "Accuracy:\t${formatter.format(metrics.accuracy() * 100)}")


    println(fprs.joinToString("\n"))
}

fun evaluateModel(ovrModel: OneVsRestModel, test: Dataset<Row>, indexer: StringIndexerModel): MulticlassMetrics {
    // Convert indexed labels back to original labels.
    val labelConverter = IndexToString()
            .setInputCol(predictionCol)
            .setOutputCol("predictedLabel")
            .setLabels(indexer.labels())


    val predicatePipeline = Pipeline().setStages(arrayOf(ovrModel, labelConverter))

    val cachedTest = test.cache()
    val predictions = predicatePipeline.fit(cachedTest).transform(cachedTest)

    predictions.show(3)
    // evaluate the model
    val predictionsAndLabels = predictions.select(predictionCol, labelIndexCol).toJavaRDD().map({ row ->
        Tuple2(row.getDouble(0) as Any, row.getDouble(1) as Any)
    })

    val metrics = MulticlassMetrics(predictionsAndLabels.rdd())
    return metrics
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