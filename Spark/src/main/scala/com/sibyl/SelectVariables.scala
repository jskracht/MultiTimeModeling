package com.sibyl

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.linalg.Vector

class SelectVariables(val spark: SparkSession) {

  def testCorrelation(indicatorVariable: Vector, outcomeVariable: Vector): Double = {
    Statistics.corr(spark.sparkContext.parallelize(indicatorVariable.toArray), spark.sparkContext.parallelize(outcomeVariable.toArray), "pearson")
  }

  def testCorrelation(indicatorVariable: Array[Double], outcomeVariable: Array[Double]): Double = {
    Statistics.corr(spark.sparkContext.parallelize(indicatorVariable), spark.sparkContext.parallelize(outcomeVariable), "pearson")
  }

  def buildLinearRegressionModel(dataFrame: DataFrame): LinearRegressionModel = {
    val lir = new LinearRegression().setFeaturesCol("features").setLabelCol("label").setRegParam(0.0).setElasticNetParam(0.0).setMaxIter(100).setTol(1E-6)
    lir.fit(dataFrame)
  }

  def buildRandomForestModel(dataFrame: DataFrame): RandomForestRegressionModel = {
    val randomForestRegressor = new RandomForestRegressor()
    randomForestRegressor.fit(dataFrame)
  }
}