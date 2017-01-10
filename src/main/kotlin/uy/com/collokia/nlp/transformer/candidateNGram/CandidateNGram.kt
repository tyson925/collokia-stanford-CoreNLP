package uy.com.collokia.nlp.transformer.candidateNGram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer

const val candidateNgramOutputColName = "candidates"

class CandidateNGram : PersistableUnaryTransformer<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>, CandidateNGram>() {

    override fun createTransformFunc(): Function1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>> {
        return ExtractFunction()
    }

    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String? {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }
}