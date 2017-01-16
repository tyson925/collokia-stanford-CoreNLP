package opennlp.tools.tokenize

import opennlp.model.MaxentModel
import opennlp.tools.dictionary.Dictionary
import opennlp.tools.tokenize.lang.Factory
import opennlp.tools.util.Span
import java.util.*
import java.util.regex.Pattern

/**
 * A Tokenizer for converting raw text into separated tokens.  It uses
 * Maximum Entropy to make its decisions.  The features are loosely
 * based off of Jeff Reynar's UPenn thesis "Topic Segmentation:
 * Algorithms and Applications.", which is available from his
 * homepage: <http://www.cis.upenn.edu/~jcreynar>.
 * <p>
 * This tokenizer needs a statistical model to tokenize a text which reproduces
 * the tokenization observed in the training data used to create the model.
 * The {@link TokenizerModel} class encapsulates the model and provides
 * methods to create it from the binary representation.
 * <p>
 * A tokenizer instance is not thread safe. For each thread one tokenizer
 * must be instantiated which can share one <code>TokenizerModel</code> instance
 * to safe memory.
 * <p>
 * To train a new model {{@link #train(String, ObjectStream, boolean, TrainingParameters)} method
 * can be used.
 * <p>
 * Sample usage:
 * <p>
 * <code>
 * InputStream modelIn;<br>
 * <br>
 * ...<br>
 * <br>
 * TokenizerModel model = TokenizerModel(modelIn);<br>
 * <br>
 * Tokenizer tokenizer = new TokenizerME(model);<br>
 * <br>
 * String tokens[] = tokenizer.tokenize("A sentence to be tokenized.");
 * </code>
 *
 * @see Tokenizer
 * @see TokenizerModel
 * @see TokenSample
 */
class TokenizerThreadsafeME: Tokenizer {
    override fun tokenize(s: String): Array<String> {
        return Span.spansToStrings(tokenizePos(s), s)
    }
    override fun tokenizePos(s: String?): Array<out Span> {
        return this.tokenizePosAndProb(s!!).first
    }

    companion object {
        /**
         * Constant indicates a token split.
         */
        val SPLIT = "T"

        /**
         * Constant indicates no token split.
         */
        val NO_SPLIT = "F"

        /**
         * Alpha-Numeric Pattern
         * @deprecated As of release 1.5.2, replaced by {@link Factory#getAlphanumeric(String)}
         */
        @Deprecated("As of release 1.5.2, replaced by {@link Factory#getAlphanumeric(String)}")
        val alphaNumeric = Pattern.compile(Factory.DEFAULT_ALPHANUMERIC)

        private fun getAbbreviations(abbreviations: Dictionary?): Set<String> {
            if (abbreviations == null) {
                return Collections.emptySet<String>()!!
            }
            return abbreviations.asStringSet()!!
        }


    }


    private val alphanumeric: Pattern

    /**
     * The maximum entropy model to use to evaluate contexts.
     */
    private val model: MaxentModel

    /**
     * The context generator.
     */
    private val cg: TokenContextGenerator

    /**
     * Optimization flag to skip alpha numeric tokens for further
     * tokenization
     */
    private val useAlphaNumericOptimization: Boolean

    /**
     * List of probabilities for each token returned from a call to
     * <code>tokenize</code> or <code>tokenizePos</code>.
     */
    // private var tokProbs:List<Double>

    //private var newTokens:List<Span>

    constructor(model: TokenizerModel) {
        val factory = model.factory
        this.alphanumeric = factory.alphaNumericPattern
        this.cg = factory.contextGenerator
        this.model = model.maxentModel
        this.useAlphaNumericOptimization = factory.isUseAlphaNumericOptmization

    }

    /**
     * @deprecated use {@link TokenizerFactory} to extend the Tokenizer
     *             functionality
     */
    constructor(model: TokenizerModel, factory: Factory) {
        val languageCode = model.language

        this.alphanumeric = factory.getAlphanumeric(languageCode)
        this.cg = factory.createTokenContextGenerator(languageCode,
                getAbbreviations(model.abbreviations))

        this.model = model.maxentModel
        useAlphaNumericOptimization = model.useAlphaNumericOptimization()
    }


    /**
     * Returns the probabilities associated with the most recent
     * calls to {@link TokenizerME#tokenize(String)} or {@link TokenizerME#tokenizePos(String)}.
     *
     * @return probability for each token returned for the most recent
     * call to tokenize.  If not applicable an empty array is
     * returned.
     * we use tokProbs.toTypedArray() here
     */

    /**
     * Tokenizes the string.
     *
     * @param d  The string to be tokenized.
     *
     * @return   A span array containing individual tokens as elements.
     */
    fun tokenizePosAndProb(d: String): Pair<Array<Span>, ArrayList<Double>> {
        val tokens = WhitespaceTokenizer.INSTANCE.tokenizePos(d)
        val newTokens = ArrayList<Span>()
        val tokProbs = ArrayList<Double>(50)

        for (i in 0 until tokens.size) {
            val s = tokens[i]
            val tok = d.substring(s.start, s.end)
            // Can't tokenize single characters
            if (tok.length < 2) {
                newTokens.add(s)
                tokProbs.add(1.toDouble())
            } else if (useAlphaNumericOptimization && alphanumeric.matcher(tok).matches()) {
                newTokens.add(s)
                tokProbs.add(1.toDouble())
            } else {
                var start = s.start
                val end = s.end
                val origStart = s.start
                var tokenProb = 1.0
                for (j in (origStart + 1) until end) {
                    val probs =
                            model.eval(cg.getContext(tok, j - origStart))
                    val best = model.getBestOutcome(probs)
                    tokenProb *= probs[model.getIndex(best)]
                    if (best == TokenizerME.SPLIT) {
                        newTokens.add(Span(start, j))
                        tokProbs.add(tokenProb)
                        start = j
                        tokenProb = 1.0
                    }
                }
                newTokens.add(Span(start, end))
                tokProbs.add(tokenProb)
            }
        }

        val spans = newTokens.toTypedArray()

        return Pair(spans, tokProbs)
    }

}
