package com.sibyl

import java.sql.Timestamp
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}
import java.util.Date

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.io.Source
import scala.util.parsing.json._
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

class FREDDataLoad(val spark: SparkSession) {

  //Gets HTTP response from URL
  @throws(classOf[java.io.IOException])
  @throws(classOf[java.net.SocketTimeoutException])
  def get(url: String,
          connectTimeout: Int = 5000,
          readTimeout: Int = 5000,
          requestMethod: String = "GET"): String = {
    import java.net.{HttpURLConnection, URL}
    val connection = new URL(url).openConnection.asInstanceOf[HttpURLConnection]
    connection.setConnectTimeout(connectTimeout)
    connection.setReadTimeout(readTimeout)
    connection.setRequestMethod(requestMethod)
    val inputStream = connection.getInputStream
    val content = Source.fromInputStream(inputStream).mkString
    if (inputStream != null) inputStream.close()
    content
  }

  //Convert FRED YYYY-MM-DD formated string to date
  private def convertStringToDate(date: String): Timestamp = {
    Timestamp.from(ZonedDateTime.of(LocalDateTime.parse(date + "T00:00:00"), ZoneId.systemDefault()).toInstant)
  }

  //Convert FRED value column to double
  def parseDoubleOrZero(value: String): Double = {
    try {
      value.toDouble
    } catch {
      case _: Exception => null.asInstanceOf[Double]
    }
  }

  //Read in seriesIDs from file, pull data for each ID, and save each observation as a row
  def getRowData(seriesListPath: String): ListBuffer[Row] = {
    var rowData = ListBuffer[Row]()
    for (series <- getSeriesIDs(seriesListPath)) {
      println("Fetching Series: " + series)

      val content = get("https://api.stlouisfed.org/fred/series/observations?series_id=%s&api_key=%s&file_type=json".
        format(series, "0ed28d55d3e9655415b8e31652c8a952")
      )
      val respFromFRED = JSON.parseFull(content)
      val observations = respFromFRED.get.asInstanceOf[Map[String, Any]]("observations").asInstanceOf[List[Any]]

      for (singleObservation <- observations) {
        val date = convertStringToDate(singleObservation.asInstanceOf[Map[String, Date]]("date").asInstanceOf[String])
        val value = parseDoubleOrZero(singleObservation.asInstanceOf[Map[String, Double]]("value").asInstanceOf[String])
        rowData += Row(date, series, value)
      }
    }
    rowData
  }

  def getSeriesIDs(seriesListPath: String): List[String] = {
    Source.fromFile(seriesListPath).getLines().toList
  }

  //Define schema to create dataframe from RDD
  def createObservationsDataFrameFromRDD(rowData: RDD[Row]): DataFrame = {
    val fields = Seq(
      StructField("date", TimestampType, nullable = true),
      StructField("series", StringType, nullable = true),
      StructField("rawValue", DoubleType, nullable = true)
    )
    val schema = StructType(fields)
    spark.sqlContext.createDataFrame(rowData, schema)
  }
}