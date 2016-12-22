package uy.com.collokia.nlp.transformer.candidateNGram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer

class CandidateNGram : PersistableUnaryTransformer<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>, CandidateNGram>() {

    override fun createTransformFunc(): Function1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<String>> {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String? {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }
}