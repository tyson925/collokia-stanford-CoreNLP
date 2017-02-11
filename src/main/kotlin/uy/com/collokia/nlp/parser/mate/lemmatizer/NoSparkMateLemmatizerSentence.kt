@file:Suppress("unused")

package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.data.SentenceData09
import scala.collection.JavaConverters
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocument
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzedBow
import uy.com.collokia.common.data.dataClasses.corpus.SimpleDocumentAnalyzedSentences
import uy.com.collokia.common.utils.nospark.NoSparkTransformer1to1
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.parser.NLPToken
import java.io.Serializable
import java.util.*

class SimpleDocumentAnalyzedGrammar(
        id: String ="",
        title: String="",
        url: String="",
        content: String="",
        tags: List<String> =listOf(),
        category: String ="",
        var analyzedContent: List<List<scala.collection.mutable.Map<String,String>>> = listOf()) : SimpleDocument(
        id,
        title,
        url,
        content,
        tags,
        category
)


class NoSparkMateLemmatizerGrammar(val language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<SimpleDocumentAnalyzedSentences, SimpleDocumentAnalyzedGrammar, List<List<String>>, List<List<scala.collection.mutable.Map<String,String>>>>(
        SimpleDocumentAnalyzedSentences::analyzedContent,
        SimpleDocumentAnalyzedGrammar::analyzedContent
), Serializable {

    val lemmatizerWrapper by lazy{
        val lemmatizerModel = if (language == LANGUAGE.ENGLISH) englishLemmatizerModelName else spanishLemmatizerModelName
        val options = arrayOf("-model", lemmatizerModel)
        LematizerWrapper(options)
    }

    override fun transfromData(dataIn: List<List<String>>): List<List<scala.collection.mutable.Map<String,String>>> {
        val sentences = dataIn
        val lemmatizer = lemmatizerWrapper.get()
        val results = ArrayList<List<scala.collection.mutable.Map<String, String>>>(sentences.size)
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
            }

            results.add(sentenceNum, lemmatizedTokens)
        }
        return results
    }
}

class NoSparkMateLemmatizerSentences(language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<SimpleDocumentAnalyzedSentences, SimpleDocumentAnalyzedSentences, List<List<String>>, List<List<String>>>(
        SimpleDocumentAnalyzedSentences::analyzedContent,
        SimpleDocumentAnalyzedSentences::analyzedContent
), Serializable {
    val lemmatizer = NoSparkMateLemmatizerGrammar(language)
    override fun transfromData(dataIn: List<List<String>>): List<List<String>> {
        return lemmatizer.transfromData(dataIn).map { s->
            s.map{ t->
                t[NLPToken::lemma.name].get()
            }
        }
    }
}

class NoSparkMateLemmatizerBow(language: LANGUAGE = LANGUAGE.ENGLISH
) : NoSparkTransformer1to1<SimpleDocumentAnalyzedSentences, SimpleDocumentAnalyzedBow, List<List<String>>, List<String>>(
        SimpleDocumentAnalyzedSentences::analyzedContent,
        SimpleDocumentAnalyzedBow::analyzedContent
), Serializable {
    val lemmatizer = NoSparkMateLemmatizerGrammar(language)
    override fun transfromData(dataIn: List<List<String>>): List<String> {
        return lemmatizer.transfromData(dataIn).map { s->
            s.map{ t->
                t[NLPToken::lemma.name].get()
            }
        }.flatten()
    }
}
