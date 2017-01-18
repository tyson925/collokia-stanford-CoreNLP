package uy.com.collokia.nlp.parser.mate.tagger

import is2.tag.Options
import is2.tag.Tagger
import java.io.Serializable
import java.util.*

class TaggerWrapper(private val props: Array<String>) : Serializable {

    companion object {
        @Volatile @Transient private var taggerMap = HashMap<String, Tagger>()

        @Synchronized fun getTagger(props: Array<String>): Tagger {
            val optsTagger = Options(props)
            return taggerMap[props.joinToString("_")].let {
                it ?: Tagger(optsTagger).let {
                    taggerMap.put(props.joinToString("_"), it)
                    it
                }
            }
        }
    }

    @Synchronized fun get(): Tagger {
        return getTagger(props)
    }

}