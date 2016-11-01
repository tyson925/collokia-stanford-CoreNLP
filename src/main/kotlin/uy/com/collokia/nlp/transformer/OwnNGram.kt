package uy.com.collokia.nlp.transformer

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray


class OwnNGram : PersistableUnaryTransformer<WrappedArray<String>, Array<String>, OwnNGram>() {
    override fun createTransformFunc(): Function1<WrappedArray<String>,Array<String>>? {
        return ConvertFunction()
    }

    override fun outputDataType(): DataType? {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String? {
        throw UnsupportedOperationException()
    }


}

