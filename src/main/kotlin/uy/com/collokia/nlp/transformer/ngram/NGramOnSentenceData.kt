package uy.com.collokia.nlp.transformer.ngram

import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.DataTypes
import scala.Function1
import scala.collection.mutable.WrappedArray
import uy.com.collokia.nlp.parser.nlpTokenType
import uy.com.collokia.nlp.transformer.PersistableUnaryTransformer

const val NGRAM_CONTENT_COL = "ngram_content"

class NGramOnSentenceData :
        PersistableUnaryTransformer<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>,
                Array<Array<Map<String, String>>>, NGramOnSentenceData>() {

    override fun createTransformFunc(): Function1<WrappedArray<WrappedArray<scala.collection.immutable.Map<String, String>>>,
            Array<Array<Map<String, String>>>> {
        return ConvertFunctionOnSentenceData()
    }

    override fun outputDataType(): DataType {
        return DataTypes.createArrayType(DataTypes.createArrayType(nlpTokenType()))
    }

    override fun uid(): String {
        //return UUID.randomUUID().toString()
        return "uid1111111"
    }


}
