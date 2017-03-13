package com.sibyl

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}

class CleanData(val spark: SparkSession) {

  //Outputs a dataframe with an added column that is vectorized and normalized
  def normalizeData(dataFrame: DataFrame, currentColumnName: String, newColumnName: String): DataFrame = {
    val vectorize = udf((rawValue: Double) => Vectors.dense(Array(rawValue)))
    val vectorizedData = dataFrame.withColumn(currentColumnName, vectorize(dataFrame(currentColumnName)))
    val scaler = new MinMaxScaler()
      .setInputCol(currentColumnName)
      .setOutputCol(newColumnName)
      .setMax(1)
      .setMin(0)

    scaler.fit(vectorizedData).transform(vectorizedData)
  }

  //Convert dataframe with vectors of length 1 into doubles
  def devectorizeData(dataFrame: DataFrame, columnName: String): DataFrame = {
    val devectorize = udf { value: DenseVector => value(0) }
    dataFrame.withColumn(columnName, devectorize(dataFrame(columnName)))
  }

}