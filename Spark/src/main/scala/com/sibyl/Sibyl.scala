package com.sibyl

import java.time.{LocalDateTime, ZoneId, ZonedDateTime}

import com.cloudera.sparkts.{DateTimeIndex, DayFrequency, TimeSeriesRDD}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
object Sibyl extends App {
  val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  import spark.implicits._

  //Read in Test Data
  val dataFromParquet = spark.sqlContext.
    read.parquet("data/testData")

  //Read in Data from FRED API
  val FREDdataLoad = new FREDDataLoad(spark)
//  val rowData = FREDdataLoad.getRowData("data/testSeries")
//  val rowRDD = spark.sparkContext.parallelize(rowData)
//  val observationsDataFrame = FREDdataLoad.createObservationsDataFrameFromRDD(rowRDD)

  //Save data to parquet for quick load
  //observationsDataFrame.write.parquet("data/testData")

  val cleanData = new CleanData(spark)
  var normalizedDataFrame = cleanData.normalizeData(dataFromParquet, "rawValue", "value")
  normalizedDataFrame = cleanData.devectorizeData(normalizedDataFrame, "value")

  val zone = ZoneId.systemDefault()
  val dateTimeIndex = DateTimeIndex.uniformFromInterval(
    ZonedDateTime.of(LocalDateTime.parse("1960-01-01T00:00:00"), zone),
    ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone), new DayFrequency(1))
  val timeSeriesRDD = TimeSeriesRDD.timeSeriesRDDFromObservations(dateTimeIndex, normalizedDataFrame,
    "date", "series", "value")
  timeSeriesRDD.cache()

  //Fill in null values using linear interpolation
  var filledTimeSeriesRDD = timeSeriesRDD.fill("linear")
  filledTimeSeriesRDD = filledTimeSeriesRDD.removeInstantsWithNaNs()

  val selectVariables = new SelectVariables(spark)
  for (series <- FREDdataLoad.getSeriesIDs("data/testSeries")) {
    if (series != "RECPROUSM156N") {
      val correlation = selectVariables.testCorrelation(filledTimeSeriesRDD.findSeries("RECPROUSM156N"), filledTimeSeriesRDD.findSeries(series))
      if (correlation > 0.75 || correlation < -0.75){
        println(series + " has correlation of: " + correlation)
      }
    }
  }

  //TODO Temporary
  val slicedTimeSeriesRDD = filledTimeSeriesRDD.slice(ZonedDateTime.of(LocalDateTime.parse("1996-01-01T00:00:00"), zone), ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone))

  val instantsDataFrame = slicedTimeSeriesRDD.toInstantsDataFrame(spark.sqlContext, 1)
  instantsDataFrame.show()

  val labeledPointsDataFrame = convertInstantsDataFrameToLabeledPointsDataFrame(instantsDataFrame)
  labeledPointsDataFrame.show()

  val linearRegressionModel = selectVariables.buildLinearRegressionModel(labeledPointsDataFrame)
  println(s"Coefficients: ${linearRegressionModel.coefficients} Intercept: ${linearRegressionModel.intercept}")
  val trainingSummary = linearRegressionModel.summary
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

  //Train Random Forest Regressor Model
  val randomForestRegressionModel = selectVariables.buildRandomForestModel(labeledPointsDataFrame)
  val output = randomForestRegressionModel.transform(labeledPointsDataFrame)
  output.show()

  def convertInstantsDataFrameToLabeledPointsDataFrame(dataFrame: DataFrame): DataFrame = {
    val ignored = List("instant", "00XALCATM086NEST")
    val featInd = instantsDataFrame.columns.diff(ignored).map(instantsDataFrame.columns.indexOf(_))
    val targetInd = instantsDataFrame.columns.indexOf("00XALCATM086NEST")
     instantsDataFrame.rdd.map(r => LabeledPoint(
      r.getDouble(targetInd), Vectors.dense(featInd.map(r.getDouble))
    )).toDF()
  }
}