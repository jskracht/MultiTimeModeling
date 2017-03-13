package com.sibyl

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.regression.LinearRegressionModel

class SelectVariables(val spark: SparkSession) {

  def testCorrelation(indicatorVariable: Vector, outcomeVariable: Vector): Double = {
    Statistics.corr(spark.sparkContext.parallelize(indicatorVariable.toArray), spark.sparkContext.parallelize(outcomeVariable.toArray), "pearson")
  }

  def buildLinearRegressionModel(dataFrame: DataFrame): LinearRegressionModel = {
    val lir = new LinearRegression().setFeaturesCol("features").setLabelCol("label").setRegParam(0.0).setElasticNetParam(0.0).setMaxIter(100).setTol(1E-6)
    lir.fit(dataFrame)
  }

  def buildRandomForestModel(dataFrame: DataFrame): RandomForestClassificationModel = {
    val randomForestClassifier = new RandomForestClassifier()
    randomForestClassifier.fit(dataFrame)
  }
}