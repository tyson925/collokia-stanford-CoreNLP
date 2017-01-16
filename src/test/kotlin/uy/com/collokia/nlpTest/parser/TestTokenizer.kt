package uy.com.collokia.nlpTest.parser

import uy.com.collokia.common.utils.rdd.convertRDDToDF
import uy.com.collokia.common.utils.rdd.getLocalSparkContext
import uy.com.collokia.common.utils.rdd.getLocalSparkSession
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.openNLP.OpenNlpTokenizer
import java.io.Serializable

data class TestDocument(var id : String, var content : String) : Serializable

class TestTokenizer() : Serializable {

    companion object {
        const val TEXT = "La jornada se realiza con motivo del 30 ° aniversario del Área Educación de FLACSO. Por este motivo invitan a aquellos investigadores en formación que quieran dar a conocer sus trabajos, a enviar sus resúmenes."

        @JvmStatic fun main(args: Array<String>) {
            val jsc = getLocalSparkContext("test")
            val sparkSession = getLocalSparkSession("test")

            val testList = listOf(TestDocument("1", TEXT))
            val testCorpus = jsc.parallelize(testList).convertRDDToDF(sparkSession)
            val tokenizer = OpenNlpTokenizer(sparkSession, TestDocument::content.name, LANGUAGE.SPANISH, isOutputRaw = false)

            val tokenized = tokenizer.transform(testCorpus)!!

            tokenized.show(false)

        }
    }



}
