@file:Suppress("RemoveForLoopIndices")

package uy.com.collokia.nlp.parser.stanford.coreNLP

import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.pipeline.Annotation
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConversions
import java.io.Serializable
import java.util.*
import kotlin.properties.Delegates


//public data class ParsedSoPage(var id: String, var parsedContent: String, var title: String, var codes: String,
//                               var imports: List<String>, var tags: List<String>, var category: String) : Serializable

data class ParsedSoPage(var id: String, var parsedContent: String, var title: String, var codes: String,
                        var imports: String, var tags: String, var category: String) : Serializable


class CoreNLPSO : Transformer, Serializable {

    var wrapper: StanfordCoreNLPWrapper
    var sparkSession: SparkSession
    var annotations: String
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
        inputColName = "content"
    }


    override fun copy(p0: ParamMap): Transformer {
        return CoreNLPSO(sparkSession, annotations)
    }

    override fun transformSchema(schema: StructType?): StructType? {

        var res = schema?.add(DataTypes.createStructField("parsedContent", DataTypes.StringType, false))
        res = res?.add(DataTypes.createStructField("id", DataTypes.StringType, false))
        res = res?.add(DataTypes.createStructField("title", DataTypes.StringType, false))
        res = res?.add(DataTypes.createStructField("category", DataTypes.StringType, false))
        res = res?.add(DataTypes.createStructField("imports", DataTypes.createArrayType(DataTypes.StringType), false))
        res = res?.add(DataTypes.createStructField("codes", DataTypes.StringType, false))
        res = res?.add(DataTypes.createStructField("tags", DataTypes.createArrayType(DataTypes.StringType), false))

        return res
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {

        return dataset?.let {
            val colums = listOf(dataset.col("id"), dataset.col("title"), dataset.col("codes"), dataset.col("imports"), dataset.col("tags"),
                    dataset.col("category"), dataset.col(inputColName))
            val columsSeq = JavaConversions.asScalaBuffer(colums)
            val selectContent = dataset.select(columsSeq)
            val beanEncoder = Encoders.bean(ParsedSoPage::class.java)

            val parsedContent = selectContent.map({ text ->

                //val parsedContent = selectContent?.javaRDD()?.map { text ->
                val content = text.getString(text.fieldIndex(inputColName))
                val doc = Annotation(content)
                wrapper.get()?.annotate(doc)
                val sentences = doc.get(CoreAnnotations.SentencesAnnotation::class.java)
                val numberOfSentences = sentences.size
                //val tokens = ArrayList<List<String>>(numberOfSentences)
                val tokens = LinkedList<String>()
                val poses = LinkedList<String>()
                val lemmas = LinkedList<String>()

                for ((sentenceIndex, sentence) in sentences.withIndex()) {
                    val documentTokensList = sentence.get(CoreAnnotations.TokensAnnotation::class.java)

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

                }
                val id = text.getString(text.fieldIndex("id"))
                val title = text.getString(text.fieldIndex("title"))
                val codes = text.getString(text.fieldIndex("codes"))
                val imports = text.getList<String>(text.fieldIndex("imports"))
                //val tags = text.getList<String>(text.fieldIndex("tags")).map { tag -> tag.replace(" ", "_") }
                val tags = text.getString(text.fieldIndex("tags"))
                val category = text.getString(text.fieldIndex("category"))
                ParsedSoPage(id, lemmas.joinToString(" "), title, codes, imports.joinToString("\n"), tags, category)

            }, beanEncoder)

            parsedContent?.toDF()
        } ?: dataset?.toDF()

    }

    override fun uid(): String? {
        return UUID.randomUUID().toString()
    }

    fun setInputCol(inputCol: String): CoreNLPSO {
        inputColName = inputCol
        return this
    }
}