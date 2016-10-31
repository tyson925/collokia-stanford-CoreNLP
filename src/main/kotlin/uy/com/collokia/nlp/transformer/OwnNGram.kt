package uy.com.collokia.nlp.transformer

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import kotlin.reflect.jvm.reflect

class OwnNGram : UnaryTransformer<WrappedArray<String>, Array<String>, OwnNGram>(), DefaultParamsWritable {

    override fun write(): MLWriter {

        return MLWritable::write.call("OwnNGram")
    }

    override fun save(path: String) {
        MLWritable::save.call(path)
    }


    override fun createTransformFunc(): Function1<WrappedArray<String>, Array<String>> {
        return ConvertFunction()
    }


    override fun outputDataType(): DataType {
        return DataTypes.createArrayType(DataTypes.StringType)
    }

    override fun uid(): String {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }
}

