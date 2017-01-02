package uy.com.collokia.nlp.parser.mate.parser

import is2.parser.Parser
import is2.tag.Options
import java.io.Serializable


class ParserWrapper(private val props : Array<String>) : Serializable {

    @Transient private var parser: Parser? = null

    fun get() : Parser {

        return if (parser == null) {
            val optsTagger = Options(props)
            parser = Parser(optsTagger)
            parser!!
        } else {
            parser!!
        }
    }
}