
package uy.com.collokia.nlp.parser.mate.lemmatizer

import is2.lemmatizer.Lemmatizer
import is2.lemmatizer.Options
import java.io.Serializable
import java.util.*

class LematizerWrapper(private val props : Array<String>) : Serializable {

    companion object{
        @Volatile @Transient private var lemmatizerMap = HashMap<String, Lemmatizer>()

        @Synchronized fun getLemmatizer(props: Array<String>): Lemmatizer {
            val optsLemmatizer = Options(props)
            return lemmatizerMap[props.joinToString("_")].let {
                it ?: Lemmatizer(optsLemmatizer).let {
                    lemmatizerMap.put(props.joinToString("_"), it)
                    it
                }
            }
        }
    }

    @Synchronized  fun get() : Lemmatizer {

        return getLemmatizer(props)
    }
}
