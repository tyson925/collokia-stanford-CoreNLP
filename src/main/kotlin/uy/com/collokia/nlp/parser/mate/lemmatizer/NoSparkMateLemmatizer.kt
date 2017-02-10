@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.data.SentenceData09
import scala.collection.JavaConverters
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzed
import uy.com.collokia.common.utils.nospark.NoSparkParamsDef
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.NLPToken
import java.io.Serializable
import java.util.*


class NoSparkMateLemmatizer(val language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<Array<Array<String>>, Array<Array<scala.collection.mutable.Map<String,String>>>>(
        NoSparkParamsDef(SimpleDocument::content.name, Array<Array<String>>::class.java),
        NoSparkParamsDef(SimpleDocumentAnalyzed::analyzedContent.name, Array<Array<scala.collection.mutable.Map<String,String>>>::class.java)
), Serializable {

    val lemmatizerWrapper by lazy{
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        LematizerWrapper(options)
    }

    override fun transfromData(dataIn: Array<Array<String>>): Array<Array<scala.collection.mutable.Map<String,String>>> {
        val sentences = dataIn
        val lemmatizer = lemmatizerWrapper.get()
        val results = ArrayList<Array<scala.collection.mutable.Map<String, String>>>(sentences.size)
        (0..sentences.size - 1).forEach { sentenceNum ->

            val tokens = sentences[sentenceNum]

            val sentenceArray = arrayOfNulls<String>(tokens.size + 1) // according to the "root"

            sentenceArray[0] = "<root>"

            (0..tokens.size - 1).forEach { i -> sentenceArray[i + 1] = tokens[i] }

            val lemmatized = SentenceData09()
            lemmatized.init(sentenceArray)
            lemmatizer.apply(lemmatized)
            val lemmas = lemmatized.plemmas

            val contentIndex = results.map { sentence -> sentence.size }.sum()

            val lemmatizedTokens = sentenceArray.mapIndexed { tokenIndex, token ->

                val values = JavaConverters.mapAsScalaMapConverter(mapOf(
                        NLPToken::index.name to tokenIndex.toString(),
                        NLPToken::token.name to (token ?: ""),
                        NLPToken::lemma.name to lemmas[tokenIndex],
                        NLPToken::indexInContent.name to (contentIndex + tokenIndex).toString()
                )).asScala()
                values
            }.toTypedArray()

            results.add(sentenceNum, lemmatizedTokens)
        }
        return results.toTypedArray()
    }
}

class NoSparkMateLemmatizerRaw(language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<Array<Array<String>>, Array<Array<String>>>(
        NoSparkParamsDef(SimpleDocument::content.name, Array<Array<String>>::class.java),
        NoSparkParamsDef(SimpleDocumentAnalyzed::analyzedContent.name, Array<Array<String>>::class.java)
), Serializable {
    val lemmatizer = NoSparkMateLemmatizer(language)
    override fun transfromData(dataIn: Array<Array<String>>): Array<Array<String>> {
        return lemmatizer.transfromData(dataIn).map { s->
            s.map{ t->
                t[NLPToken::lemma.name].get()
            }.toTypedArray()

        }.toTypedArray()
    }

}

class NoSparkMateLemmatizerRawRaw(language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<Array<Array<String>>, Array<String>>(
        NoSparkParamsDef(SimpleDocument::content.name, Array<Array<String>>::class.java),
        NoSparkParamsDef(SimpleDocumentAnalyzed::analyzedContent.name, Array<String>::class.java)
), Serializable {
    val lemmatizer = NoSparkMateLemmatizer(language)
    override fun transfromData(dataIn: Array<Array<String>>): Array<String> {
        return lemmatizer.transfromData(dataIn).map { s->
            s.map{ t->
                t[NLPToken::lemma.name].get()
            }

        }.flatten().toTypedArray()
    }

}