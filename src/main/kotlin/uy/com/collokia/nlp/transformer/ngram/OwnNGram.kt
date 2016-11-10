package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer


class OwnNGram : PersistableUnaryTransformer<WrappedArray<String>, Array<String>, OwnNGram>() {
    override fun createTransformFunc(): Function1<WrappedArray<String>, Array<String>>? {
        return ConvertFunction()
    }

    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String? {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }


}


