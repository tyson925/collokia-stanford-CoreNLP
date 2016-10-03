package uy.com.collokia.nlp.parser.stanford.coreNLP


import java.io.Serializable
import java.util.*
import edu.stanford.nlp.pipeline.StanfordCoreNLP

class StanfordCoreNLPWrapper(private val props: Properties) : Serializable {

    @Transient private var coreNLP: StanfordCoreNLP? = null

    fun get() : StanfordCoreNLP? {

        if (coreNLP == null) {
            coreNLP = StanfordCoreNLP(props)
        }
        return coreNLP
    }
}
