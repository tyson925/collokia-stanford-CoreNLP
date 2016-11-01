
package uy.com.collokia.nlp.transformer

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}



abstract class PersistableUnaryTransformer[IN, OUT, T<:UnaryTransformer[IN, OUT, T]] extends UnaryTransformer[IN, OUT, T] with DefaultParamsReadable[T] with DefaultParamsWritable




