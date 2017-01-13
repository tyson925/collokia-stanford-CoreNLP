package uy.com.collokia.nlp.parser.mate.tagger

import is2.tag.Options
import is2.tag.Tagger
import java.io.Serializable

class TaggerWrapper(private val props : Array<String>) : Serializable {

    companion object {
        @Transient private var tagger: Tagger? = null
    }

    fun get() : Tagger {
        return if (tagger == null) {
            val optsTagger = Options(props)
            tagger = Tagger(optsTagger)
            tagger!!
        } else {
            tagger!!
        }
    }

}