package opennlp.tools.sentdetect

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */




import opennlp.model.MaxentModel
import opennlp.tools.dictionary.Dictionary
import opennlp.tools.sentdetect.lang.Factory
import opennlp.tools.util.Span
import opennlp.tools.util.StringUtil
import java.util.*

/**
 * A sentence detector for splitting up raw text into sentences.
 * <p>
 * A maximum entropy model is used to evaluate the characters ".", "!", and "?" in a
 * string to determine if they signify the end of a sentence.
 */
class SentenceDetectorThreadsafeME : SentenceDetector {
    override fun sentPosDetect(s: String?): Array<out Span> {
        return this.sentPosDetectProbs(s!!).first
    }

    companion object {
        /**
         * Constant indicates a sentence split.
         */
        val SPLIT = "s"

        /**
         * Constant indicates no sentence split.
         */
        val NO_SPLIT = "n"

        // Note: That should be inlined when doing a re-factoring!
        private val ONE = 1.toDouble()

        private fun getAbbreviations(abbreviations: Dictionary?): Set<String> {
            if (abbreviations == null) {
                return Collections.emptySet<String>()
            }
            return abbreviations.asStringSet()
        }
    }


    /**
     * The maximum entropy model to use to evaluate contexts.
     */
    private val model: MaxentModel

    /**
     * The feature context generator.
     */
    private val cgen: SDContextGenerator

    /**
     * The {@link EndOfSentenceScanner} to use when scanning for end of sentence offsets.
     */
    private val scanner: EndOfSentenceScanner;

    /**
     * The list of probabilities associated with each decision.
     */


    private val useTokenEnd: Boolean

    /**
     * Initializes the current instance.
     *
     * @param model the {@link SentenceModel}
     */
    constructor(model: SentenceModel) {
        val sdFactory = model.factory;
        this.model = model.maxentModel;
        cgen = sdFactory.sdContextGenerator;
        scanner = sdFactory.endOfSentenceScanner;
        this.useTokenEnd = sdFactory.isUseTokenEnd;
    }

    /**
     * @deprecated Use a {@link SentenceDetectorFactory} to extend
     *             SentenceDetector functionality.
     */
    constructor(model: SentenceModel, factory: Factory) {
        this.model = model.maxentModel
        // if the model has custom EOS characters set, use this to get the context
        // generator and the EOS scanner; otherwise use language-specific defaults
        val customEOSCharacters = model.eosCharacters
        if (customEOSCharacters == null) {
            cgen = factory.createSentenceContextGenerator(model.language,
                    getAbbreviations(model.abbreviations))
            scanner = factory.createEndOfSentenceScanner(model.language)
        } else {
            cgen = factory.createSentenceContextGenerator(
                    getAbbreviations(model.abbreviations), customEOSCharacters)
            scanner = factory.createEndOfSentenceScanner(customEOSCharacters)
        }
        useTokenEnd = model.useTokenEnd();
    }


    /**
     * Detect sentences in a String.
     *
     * @param s  The string to be processed.
     *
     * @return   A string array containing individual sentences as elements.
     */
    override fun sentDetect(s: String): Array<String> {
        val sentences = sentPosDetect(s).map { span ->
            span.getCoveredText(s).toString()
        }.toTypedArray()
        return sentences
    }

    private fun getFirstWS(s: String, pos: Int): Int {
        var _pos = pos
        while (_pos < s.length && !StringUtil.isWhitespace(s[_pos])) {
            _pos++
        }
        return _pos
    }

    private fun getFirstNonWS(s: String, pos: Int): Int {
        var _pos = pos
        while (_pos < s.length && StringUtil.isWhitespace(s[_pos])) {
            _pos++
        }

        return _pos
    }

    /**
     * Detect the position of the first words of sentences in a String.
     *
     * @param s  The string to be processed.
     * @return   A integer array containing the positions of the end index of
     *          every sentence
     *
     */
    fun sentPosDetectProbs(s: String): Pair<Array<Span>, ArrayList<Double>> {
        val sentProbs = ArrayList<Double>()
        val sb = StringBuffer(s)
        val enders = scanner.getPositions(s)
        val positions = ArrayList<Integer>(enders.size)

        var index = 0
        for (i in 0 until enders.size) {
            val end = enders.size
            val candidate = enders[i]
            val cint = candidate
            // skip over the leading parts of non-token final delimiters
            val fws = getFirstWS(s, cint + 1)
            if (i + 1 < end && enders[i + 1] < fws) {
                continue
            }

            var probs = model.eval(cgen.getContext(sb, cint))
            var bestOutcome = model.getBestOutcome(probs)

            if (bestOutcome.equals(SPLIT) && isAcceptableBreak(s, index, cint)) {
                if (index != cint) {
                    if (useTokenEnd) {
                        positions.add(Integer(getFirstNonWS(s, getFirstWS(s, cint + 1))))
                    } else {
                        positions.add(Integer(getFirstNonWS(s, cint)))
                    }
                    sentProbs.add(probs[model.getIndex(bestOutcome)])
                }
                index = cint + 1
            }
        }

        val starts = positions.toTypedArray()

        // string does not contain sentence end positions
        if (starts.size == 0) {

            // remove leading and trailing whitespace
            var start = 0
            var end = s.length

            while (start < s.length && StringUtil.isWhitespace(s[start])) {
                start++
            }

            while (end > 0 && StringUtil.isWhitespace(s[end - 1])) {
                end--
            }

            if ((end - start) > 0) {
                sentProbs.add(1.toDouble());
                return Pair(arrayOf(Span(start, end)), sentProbs);
            } else
                return Pair(arrayOf(), sentProbs)
        }

        // Now convert the sent indexes to spans
        val leftover = starts[starts.size - 1].toInt() != s.length;
        val arraySize = if (leftover) {
            starts.size + 1
        } else {
            starts.size
        }
        val spans = Array<Span?>(arraySize) { null }
        for (si in 0 until starts.size) {
            var start: Int
            var end: Int
            if (si == 0) {
                start = 0

                while (si < starts.size && StringUtil.isWhitespace(s[start]))
                    start++
            } else {
                start = starts[si - 1].toInt()
            }
            end = starts[si].toInt()
            while (end > 0 && StringUtil.isWhitespace(s[end - 1])) {
                end--
            }
            spans[si] = Span(start, end)
        }

        if (leftover) {
            spans[spans.size - 1] = Span(starts[starts.size - 1].toInt(), s.length)
            sentProbs.add(ONE)
        }

        return Pair(spans.map { it!! }.toTypedArray(), sentProbs)
    }

    /**
     * Allows subclasses to check an overzealous (read: poorly
     * trained) model from flagging obvious non-breaks as breaks based
     * on some boolean determination of a break's acceptability.
     *
     * <p>The implementation here always returns true, which means
     * that the MaxentModel's outcome is taken as is.</p>
     *
     * @param s the string in which the break occurred.
     * @param fromIndex the start of the segment currently being evaluated
     * @param candidateIndex the index of the candidate sentence ending
     * @return true if the break is acceptable
     */
    private fun isAcceptableBreak(s: String, fromIndex: Int, candidateIndex: Int): Boolean {
        return true
    }



}
