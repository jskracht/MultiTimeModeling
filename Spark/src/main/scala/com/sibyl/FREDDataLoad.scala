package com.sibyl

import java.io._
import java.util.Date

import scala.io.Source
import scalaj.http.{Http, HttpResponse}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StructField, _}

/**
  * Created by Jesh on 11/27/16.
  */
object FREDDataLoad {
  case class Observation(date: Date, value: Double)

  def main(args: Array[String]): Unit = {
    // Session Setup (http://spark.apache.org/docs/latest/submitting-applications.html#master-urls)
    val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()

    // Load Data
    val seriesIds = Source.fromFile("data/seriesList").getLines()
    val seriesCount = seriesIds.length
    var count = 1
    for (seriesId <- Source.fromFile("data/seriesList").getLines()) {
      val series = loadSeriesFromAPI(seriesId, spark)
      println(count + " of " + seriesCount + " imported")
      count = count + 1
    }
  }

  def loadSeriesFromAPI(seriesId: String, spark: SparkSession): org.apache.spark.sql.DataFrame = {
    val response: HttpResponse[String] = Http("https://api.stlouisfed.org/fred/series/observations").param("series_id", seriesId).param("realtime_start", "1930-01-01").param("api_key", "0ed28d55d3e9655415b8e31652c8a952").param("file_type", "json").asString
    val file = new File("data/temp.json")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(response.body)
    bw.close()

    val observations = StructType(Array(
      StructField("date", TimestampType, true),
      StructField("value", StringType, true)))
    val schema = StructType(Array(
      StructField("observations", ArrayType(observations, true), true)))

    val series = spark.read.schema(schema).json("data/temp.json")
    series
  }

  def toDouble(value: String):Double = {
    try {
      return value.toDouble
    } catch {
      case e: NumberFormatException => None
    }
    null.asInstanceOf[Double]
  }
}
