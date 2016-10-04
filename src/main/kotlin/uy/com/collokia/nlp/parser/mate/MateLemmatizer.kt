package uy.com.collokia.nlp.parser.mate

import is2.data.SentenceData09
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import uy.com.collokia.nlp.parser.openNLP.TokenizedContent
import java.io.Serializable

data class LemmatizedContent(var content : String,var tokenizedContent: List<String>, var lemmatizedContent: List<String>) : Serializable

class MateLemmatizer : Transformer {

    val lemmatizerWrapper : LematizerWrapper
    var inputColName : String
    var outputColName : String


    constructor(lemmatizerModel : String = "./../MLyBigData/NLPUtils/data/mate/models/CoNLL2009-ST-English-ALL.anna-3.3.lemmatizer.model"){
        val options = arrayOf("-model",lemmatizerModel)
        lemmatizerWrapper = LematizerWrapper(options)
        this.inputColName = TokenizedContent::tokenizedContent.name
        this.outputColName = LemmatizedContent::lemmatizedContent.name
    }

    fun setInputColName(inputColName : String) : MateLemmatizer{
        this.inputColName = inputColName
        return this
    }

    override fun uid(): String {
        return "lemmatizer111111"
    }

    override fun copy(p0: ParamMap?): Transformer {
        return MateLemmatizer()
    }

    override fun transform(dataset: Dataset<*>?): Dataset<Row>? {
        return dataset?.let{
            val selectTokenizedContent = dataset.select("*")
            val beanEncoder = Encoders.bean(LemmatizedContent::class.java)


            selectTokenizedContent.map({row ->
                val tokensIndex = row.fieldIndex(inputColName)
                val tokens = row.getList<String>(tokensIndex)
                val lemmas = if (tokens.size < 110){

                    val sentenceArray = arrayOfNulls<String>(tokens.size + 1) // according to the "root"

                    sentenceArray[0] = "<root>"

                    for (i in tokens.indices) {
                        sentenceArray[i + 1] = tokens[i]
                    }

                    val lemmatized = SentenceData09()
                    lemmatized.init(sentenceArray)

                    lemmatizerWrapper.get().apply(lemmatized).plemmas.toList()

                } else {
                    listOf()
                }
                val contentIndex = row.fieldIndex(TokenizedContent::content.name)
                LemmatizedContent(row.getString(contentIndex),tokens,lemmas)

            },beanEncoder).toDF()
        } ?: dataset?.toDF()

    }

    override fun transformSchema(p0: StructType?): StructType {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }



}