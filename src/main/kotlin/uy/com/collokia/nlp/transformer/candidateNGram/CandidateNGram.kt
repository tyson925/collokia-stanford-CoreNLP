@file:Suppress("ConvertSecondaryConstructorToPrimary")

package uy.com.collokia.nlp.transformer.candidateNGram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.parser.LANGUAGE
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer

const val CANDIDATE_NGRAM_OUTPUT_COL_NAME = "candidates"

class CandidateNGram : PersistableUnaryTransformer<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>, Array<String>, CandidateNGram> {

    val language : LANGUAGE

    constructor(language: LANGUAGE){
        this.language = language
    }

    override fun createTransformFunc(): Function1<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>, Array<String>> {
        return ExtractFunction(language)
    }

    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String? {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }
}