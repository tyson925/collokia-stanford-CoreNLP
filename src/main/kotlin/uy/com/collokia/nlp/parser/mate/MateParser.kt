package uy.com.collokia.nlp.parser.mate

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

class MateParser : Transformer {

    override fun uid(): String {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun copy(p0: ParamMap?): Transformer {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transform(p0: Dataset<*>?): Dataset<Row> {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transformSchema(p0: StructType?): StructType {
        throw UnsupportedOperationException("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    constructor(){

    }
}