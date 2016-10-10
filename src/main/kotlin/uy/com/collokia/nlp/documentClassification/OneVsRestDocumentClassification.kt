package uy.com.collokia.nlp.documentClassification


import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.*
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions
import scala.Tuple2
import uy.com.collokia.common.utils.deleteIfExists
import uy.com.collokia.common.utils.machineLearning.EvaluationMetrics
import uy.com.collokia.common.utils.machineLearning.LogisticRegressionProperties
import uy.com.collokia.common.utils.machineLearning.printMatrix
import uy.com.collokia.common.utils.rdd.readDzoneDataFromJson
import uy.com.collokia.nlp.documentClassification.vtm.*
import java.text.DecimalFormat
import uy.com.collokia.common.utils.machineLearning.printMultiClassMetrics

val OVR_MODEL = "./data/model/ovrDectisonTree"
val LABELS = "./data/model/labelIndexer_2"
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

    val vtmDataPipeline = constructVTMPipeline(stopwords.value, CONTENT_VTM_VOC_SIZE, "lemmatizedContent")

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
            "lemmatizedContent",
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

    val vtmTagPipeline = constructTagVtmDataPipeline(TAG_VTM_VOC_SIZE, tagInputColName)

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

fun evaluateOneVsRestLogReg(dataset: Dataset<Row>): LogisticRegressionProperties {
    val (train, test) = dataset.randomSplit(doubleArrayOf(0.9, 0.1))
    val indexer = StringIndexerModel.load(LABELS)

    val cachedTrain = train.cache()
    val cachedTest = test.cache()

    val evaluations =
            //listOf(100, 200, 300, 600).flatMap { numIterations ->
            listOf(200).flatMap { numIterations ->
            //listOf(600).flatMap { numIterations ->
                //listOf(1E-5, 1E-6, 1E-7).flatMap { stepSize ->
                listOf(1E-7).flatMap { stepSize ->
                    //listOf(0.001,0.01,0.1,0.3,0.5,0.8,1.0).flatMap { regressionParam ->
                    listOf(0.1).flatMap { regressionParam ->
                        listOf(0.0,0.001,0.01, 0.1).flatMap { elasticNetParam ->
                        //listOf(0.0).flatMap { elasticNetParam ->
                            //listOf(true, false).flatMap { fitIntercept ->
                            listOf(true).flatMap { fitIntercept ->
                                listOf(true).map { standardization ->

                                    val oneVsRest = constructLogRegClassifier(numIterations, stepSize, fitIntercept, standardization,
                                            regressionParam, elasticNetParam)
                                    val ovrModel = oneVsRest.fit(cachedTrain)
                                    val metrics = evaluateModel(ovrModel, cachedTest, indexer)

                                    val properties = LogisticRegressionProperties(numIterations, stepSize, fitIntercept, standardization,
                                            regressionParam, elasticNetParam)
                                    println("${metrics.weightedFMeasure()}\t$properties")
                                    Tuple2(properties, metrics)
                                }
                            }
                        }
                    }
                }
            }

    val sortedEvaluations = evaluations.sortedBy({ metricsData -> metricsData._2.fMeasure(1.0) }).reversed().map { metricsData ->
        Tuple2(metricsData._1, printMultiClassMetrics(metricsData._2))
    }

    println(sortedEvaluations.joinToString("\n"))

    val bestLogRegProperties = sortedEvaluations.first()._1

    val oneVsRest = constructLogRegClassifier(bestLogRegProperties.numIterations,
            bestLogRegProperties.stepSize,
            bestLogRegProperties.fitIntercept,
            bestLogRegProperties.standardization,
            bestLogRegProperties.regParam)

    val ovrModel = oneVsRest.fit(cachedTrain)

    evaluateModelConfusionMTX(ovrModel, cachedTest)

    return bestLogRegProperties
}

fun evaluateLogRegModel(lrModel: LogisticRegressionModel): Tuple2<Double, Double> {
    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    val trainingSummary = lrModel.summary()

    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory()
    objectiveHistory.forEach({ loss -> println(loss) })

    val binarySummary = trainingSummary as BinaryLogisticRegressionSummary

    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = binarySummary.roc()
    //roc.show()
    println("areaUnderROC:\t${binarySummary.areaUnderROC()}")

    // Set the model threshold to maximize F-Measure
    val fMeasure = binarySummary.fMeasureByThreshold()

    //fMeasure.show(100, false)
    val maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where(fMeasure.col("F-Measure").`$eq$eq$eq`(maxFMeasure)).select("threshold").head().getDouble(0)

    lrModel.threshold = bestThreshold

    println("Coefficients: ${lrModel.coefficients()} Intercept: ${lrModel.intercept()}")
    println("maxFMeasure: ${fMeasure}\tthreshold: ${bestThreshold}")
    return Tuple2(maxFMeasure, bestThreshold)
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

fun constructLogRegClassifier(numIterations: Int,
                              stepSize: Double,
                              fitIntercept: Boolean,
                              standardization: Boolean,
                              regParam: Double,
                              elasticNetParam: Double = 0.0): OneVsRest {

    val logisticRegression = LogisticRegression()
            .setMaxIter(numIterations)
            .setTol(stepSize)
            .setFitIntercept(fitIntercept)
            .setStandardization(standardization)
            .setRegParam(regParam)

    if (elasticNetParam != 0.0) {
        logisticRegression.elasticNetParam = elasticNetParam
    }

    println("elasticNetParam\t${logisticRegression.elasticNetParam}")
    val oneVsRest = OneVsRest().setClassifier(logisticRegression)
            .setFeaturesCol(featureCol)
            .setLabelCol(labelIndexCol)

    return oneVsRest
}