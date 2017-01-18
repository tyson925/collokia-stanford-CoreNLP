
package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.lemmatizer.Lemmatizer
import is2.lemmatizer.Options
import java.io.Serializable

class LematizerWrapper(private val props : Array<String>) : Serializable {

    companion object{
        @Volatile @Transient private var lemmatizer: Lemmatizer? = null
    }

    @Synchronized  fun get() : Lemmatizer {

        return if (lemmatizer == null) {
            val optsLemmatizer = Options(props)
            lemmatizer = Lemmatizer(optsLemmatizer)
            lemmatizer!!
        } else {
            lemmatizer!!
        }
    }
}
