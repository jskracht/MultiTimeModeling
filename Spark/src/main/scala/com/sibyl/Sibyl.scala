package com.sibyl

import java.time.{LocalDateTime, ZoneId, ZonedDateTime}

import com.cloudera.sparkts.{DateTimeIndex, DayFrequency, TimeSeriesRDD}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object Sibyl extends App {
  val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()

  import spark.implicits._

  val dataLoad = new FREDDataLoad(spark)
  val rowData = dataLoad.getRowData("data/seriesList2")
  val rowRDD = spark.sparkContext.parallelize(rowData)
  val observationsDataFrame = dataLoad.createObservationsDataFrameFromRDD(rowRDD)

  val cleanData = new CleanData(spark)
  var normalizedDataFrame = cleanData.normalizeData(observationsDataFrame, "rawValue", "value")
  normalizedDataFrame = cleanData.devectorizeData(normalizedDataFrame, "value")

  val zone = ZoneId.systemDefault()
  val dateTimeIndex = DateTimeIndex.uniformFromInterval(
    ZonedDateTime.of(LocalDateTime.parse("1960-01-01T00:00:00"), zone),
    ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone), new DayFrequency(1))
  val timeSeriesRDD = TimeSeriesRDD.timeSeriesRDDFromObservations(dateTimeIndex, normalizedDataFrame,
    "date", "series", "value")
  timeSeriesRDD.cache()

  //Fill in null values using linear interpolation
  val filledTimeSeriesRDD = timeSeriesRDD.fill("linear")

  //TODO Temporary
  val slicedTimeSeriesRDD = filledTimeSeriesRDD.slice(ZonedDateTime.of(LocalDateTime.parse("1996-01-01T00:00:00"), zone), ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone))

  val instantsDataFrame = slicedTimeSeriesRDD.toInstantsDataFrame(spark.sqlContext, 1)

  val selectVariables = new SelectVariables(spark)
  val correlation = selectVariables.testCorrelation(slicedTimeSeriesRDD.findSeries("00XALCATM086NEST"), slicedTimeSeriesRDD.findSeries("00XALCBEM086NEST"))
  println(correlation)

  val labeledPointsDataFrame = convertInstantsDataFrameToLabeledPointsDataFrame(instantsDataFrame)

  val linearRegressionModel = selectVariables.buildLinearRegressionModel(labeledPointsDataFrame)
  println(s"Coefficients: ${linearRegressionModel.coefficients} Intercept: ${linearRegressionModel.intercept}")
  val trainingSummary = linearRegressionModel.summary
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

  //Train Random Forest Classifier Model
  val randomForestClassifierModel = selectVariables.buildRandomForestModel(labeledPointsDataFrame)
  val output = randomForestClassifierModel.transform(labeledPointsDataFrame)
  output.show()

  def convertInstantsDataFrameToLabeledPointsDataFrame(dataFrame: DataFrame): DataFrame = {
    val ignored = List("instant", "00XALCATM086NEST")
    val featInd = instantsDataFrame.columns.diff(ignored).map(instantsDataFrame.columns.indexOf(_))
    val targetInd = instantsDataFrame.columns.indexOf("00XALCATM086NEST")
     instantsDataFrame.rdd.map(r => LabeledPoint(
      r.getDouble(targetInd).round, Vectors.dense(featInd.map(r.getDouble))
    )).toDF()
  }
}