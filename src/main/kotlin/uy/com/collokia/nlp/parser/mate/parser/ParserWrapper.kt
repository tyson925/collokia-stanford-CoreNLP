package uy.com.collokia.nlp.parser.mate.parser

import is2.parser.Parser
import is2.tag.Options
import java.io.Serializable
import java.util.*


class ParserWrapper(private val props : Array<String>) : Serializable {
    companion object {
        @Volatile @Transient private var parserMap = HashMap<String,Parser>()

        @Synchronized fun getParser(props: Array<String>): Parser {
            val optsTagger = Options(props)
            return parserMap[props.joinToString("_")].let {
                it ?: Parser(optsTagger).let {
                    parserMap.put(props.joinToString("_"), it)
                    it
                }
            }
        }
    }

    @Synchronized fun get() : Parser {
        return getParser(props)
    }
}
