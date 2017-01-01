package com.sibyl

import java.io._
import java.util.Date

import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

import scala.io.Source
import scalaj.http.{Http, HttpResponse}
import com.databricks.spark.xml
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{StructField, _}

import scala.collection.mutable

/**
  * Created by Jesh on 11/27/16.
  */
object FREDDataLoad {
  case class Observation(observation: Array[String])

  def main(args: Array[String]): Unit = {
    // Session Setup (http://spark.apache.org/docs/latest/submitting-applications.html#master-urls)
    val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()

    // Load Data
    val seriesIds = Source.fromFile("data/seriesList").getLines().toList
    val seriesCount = seriesIds.length
    var count = 1
    var allSeries = loadSeriesFromAPI(seriesIds.take(1).head, spark)
    for (seriesId <- Source.fromFile("data/seriesList").getLines()) {
      if (count > 1) {
        val series = loadSeriesFromAPI(seriesId, spark)
        allSeries = allSeries.union(series)
      }
      println(count + " of " + seriesCount + " imported")
      count = count + 1
      //allSeries.show()
    }
  }

  def loadSeriesFromAPI(seriesId: String, spark: SparkSession): DataFrame = {
    val response: HttpResponse[String] = Http("https://api.stlouisfed.org/fred/series/observations").param("series_id", seriesId).param("realtime_start", "1930-01-01").param("api_key", "0ed28d55d3e9655415b8e31652c8a952").asString
    val file = new File("data/temp.xml")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(response.body)
    bw.close()

    val observations = StructType(Array(
      StructField("_date", DateType, true),
      StructField("_value", DoubleType, true)))
    val schema = StructType(Array(
      StructField("observation", ArrayType(observations, true), true)))

    val series = spark.sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "observations").option("nullValue", ".").schema(schema).load("data/temp.xml")
    for (test <- series.select("observation._value").head.toSeq.head.asInstanceOf[mutable.WrappedArray[Double]].iterator)
      {
        println(test)
      }
    series
  }
}
