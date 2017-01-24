package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer

class NGramOnSentenceData : PersistableUnaryTransformer<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<Array<Array<String>>>, NGramOnSentenceData>() {
    override fun createTransformFunc(): Function1<WrappedArray<WrappedArray<WrappedArray<String>>>, Array<Array<Array<String>>>> {
        return ConvertFunctionOnSentenceData()
    }

    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.createArrayType(DataTypes.StringType)))
    }

    override fun uid(): String? {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }


}
