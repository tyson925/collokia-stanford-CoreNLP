package uy.com.collokia.nlp.parser.stanford.coreNLP

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.*
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import java.io.Serializable
import java.util.*

data class ParsedSentenceBean(var category: String, var categoryIndex: Double, var content: String, var title: String,
                                     var labels: List<String>, var tokens: String, var poses: String, var lemmas: String,
                                     var parses: String, var ners: String) : Serializable


class CoreNLP : Transformer {

    var wrapper: StanfordCoreNLPWrapper
    var sparkSession: SparkSession
    var annotations: String
    //var inputColName: String by Delegates.notNull<String>()
    var inputColName: String? = null
    var outputCol: String? = "lemmas"

    constructor(sparkSession: SparkSession, annotations: String) {
        this.sparkSession = sparkSession
        this.annotations = annotations

        //CoreNLP("corenlp_" + UUID.randomUUID().toString().drop(12))
        val props = Properties()
        //tokenize,
        props.setProperty("annotators", "tokenize, ssplit, ${annotations}")
        wrapper = StanfordCoreNLPWrapper(props)
    }

    override fun copy(p0: ParamMap): Transformer {
        return CoreNLP(sparkSession, annotations)
    }

    override fun transformSchema(schema: StructType?): StructType? {
        //throw UnsupportedOperationException()
        var res = schema?.add(DataTypes.createStructField("tokens", DataTypes.StringType, false))
        if (annotations.contains("pos")) {
            res = res?.add(DataTypes.createStructField("poses", DataTypes.StringType, false))
        }
        if (annotations.contains("lemma")) {
            //res = res?.add(DataTypes.createStructField("lemmas", DataTypes.createArrayType(DataTypes.StringType), false))
            res = res?.add(DataTypes.createStructField("lemmas", DataTypes.StringType, false))
        }
        if (annotations.contains("parse")) {
            res = res?.add(DataTypes.createStructField("parses", DataTypes.StringType, false))
        }
        if (annotations.contains("ner")) {
            res = res?.add(DataTypes.createStructField("ners", DataTypes.StringType, false))
        }
        //res = res?.add(DataTypes.createStructField("null", DataTypes.NullType, true))

        return res
    }


    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        val isIndex = dataset?.columns()?.toList()?.contains("categoryIndex") ?: false
        val isTitle = dataset?.columns()?.toList()?.contains("title") ?: false
        val isLabels = dataset?.columns()?.toList()?.contains("labels") ?: false
        val colums = LinkedList<Column>()
        return dataset?.let {
            colums.add(dataset.col("category"))
            if (isIndex) {
                colums.add(dataset.col("categoryIndex"))
            }
            //colums.add(dataset.col("content"))
            if (isTitle) {
                colums.add(dataset.col("title"))
            }
            if (isLabels) {
                colums.add(dataset.col("labels"))
            }
            dataset.col(inputColName)

            val columsSeq = JavaConversions.asScalaBuffer(colums)

            val selectContent = dataset.select(columsSeq)

            val beanEncoder = Encoders.bean(ParsedSentenceBean::class.java)
            val parsedContent = selectContent.map({ text ->

                //val parsedContent = selectContent?.javaRDD()?.map { text ->
                val content = text.getString(text.fieldIndex(inputColName))
                val doc = Annotation(content)
                wrapper.get()?.annotate(doc)
                val sentences = doc.get(CoreAnnotations.SentencesAnnotation::class.java)
//            val numberOfSentences = sentences.size
                //val tokens = ArrayList<List<String>>(numberOfSentences)
                val tokens = LinkedList<String>()
                val poses = LinkedList<String>()
                val lemmas = LinkedList<String>()
                val parses = LinkedList<String>()
                val ners = LinkedList<String>()

                for ((sentenceIndex, sentence) in sentences.withIndex()) {
                    val documentTokensList = sentence.get(CoreAnnotations.TokensAnnotation::class.java)
                    //println(sentence)


                    val tokenArray = documentTokensList.map { token ->
                        token.get(CoreAnnotations.TextAnnotation::class.java)
                    }
                    //tokens.add(sentenceIndex,tokenArray)
                    tokens.addAll(tokenArray)

                    val posArray = documentTokensList.map { token ->
                        token.get(CoreAnnotations.PartOfSpeechAnnotation::class.java)
                    }

                    poses.addAll(posArray)

                    val lemmaArray = documentTokensList.map { token ->
                        token.get(CoreAnnotations.LemmaAnnotation::class.java)
                    }
                    lemmas.addAll(lemmaArray)

                    if (annotations.contains("parse")) {

                        val dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation::class.java)
                        //SemanticGraph dependencies = sentence.get(CoreAnnotations.CoNLLDepAnnotation.class);
                        val edgeList = dependencies.edgeListSorted()

                        val parseArray = edgeList.map { edge ->
                            edge.toString()
                        }
                        parses.addAll(parseArray)
                    }

                    if (annotations.contains("ner")) {
                        val nerArray = documentTokensList.map { token ->
                            token.get(CoreAnnotations.NamedEntityTagAnnotation::class.java)
                        }
                        ners.addAll(nerArray)
                    }
                }

                val index = if (isIndex) {
                    text.getDouble(1)
                } else {
                    0.0
                }
                val title = if (isTitle) {
                    text.getString(text.fieldIndex("title"))
                } else {
                    ""
                }

                val labels = if (isLabels) {
                    text.getList<String>(text.fieldIndex("labels"))
                } else {
                    listOf()
                }

                val res = if (!annotations.contains("parse") && !annotations.contains("ner")) {
                    outputCol = "lemmas"
                    ParsedSentenceBean(text.getString(0), index, content, title, labels, tokens.joinToString(" "), poses.joinToString(" "),
                            lemmas.joinToString(" "), "", "")
                } else if (annotations.contains("parse") && !annotations.contains("ner")) {
                    outputCol = "parses"
                    ParsedSentenceBean(text.getString(0), index, content, title, labels, tokens.joinToString(" "), poses.joinToString(" "),
                            lemmas.joinToString(" "), parses.joinToString(" "), "")
                } else if (annotations.contains("parse") && annotations.contains("ner")) {
                    outputCol = "ners"
                    ParsedSentenceBean(text.getString(0), index, content, title, labels, tokens.joinToString(" "), poses.joinToString(" "),
                            lemmas.joinToString(" "), parses.joinToString(" "), ners.joinToString(" "))
                } else {
                    ParsedSentenceBean(text.getString(0), index, content, title, labels, tokens.joinToString(" "), "", "", "", "")
                }
                res
            }, beanEncoder)
            parsedContent.toDF()
        } ?: dataset?.toDF()

    }


    override fun uid(): String? {
        return UUID.randomUUID().toString()
    }

    fun setInputCol(inputCol: String): CoreNLP {
        inputColName = inputCol
        return this
    }


}


