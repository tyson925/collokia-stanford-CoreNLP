
package uy.com.collokia.nlp.parser.mate

import is2.lemmatizer.Lemmatizer
import is2.lemmatizer.Options
import java.io.Serializable

class LematizerWrapper(private val props : Array<String>) : Serializable {


    @Transient private var lemmatizer: Lemmatizer? = null

    fun get() : Lemmatizer {

        return if (lemmatizer == null) {
            val optsLemmatizer = Options(props)
            lemmatizer = Lemmatizer(optsLemmatizer)
            lemmatizer!!
        } else {
            lemmatizer!!
        }
    }
}
