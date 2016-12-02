package com.sibyl

import java.util.Date
import scala.collection.immutable.Seq
import scala.io.Source
import scalaj.http.{Http, HttpResponse}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Jesh on 11/27/16.
  */
object FREDDataLoad {
  case class Observation(date: Date, value: Double)

  def main(args: Array[String]): Unit = {
    // Context Setup
    val conf = new SparkConf().setAppName("Sibyl").setMaster("local")
    val sc = new SparkContext(conf)

    // Load Data
    val seriesIds = Source.fromFile("data/seriesList").getLines()
    val seriesCount = seriesIds.length
    var count = 0
    val allSeries = scala.collection.mutable.Map[String, Seq[FREDDataLoad.Observation]]()
    for (seriesId <- Source.fromFile("data/seriesList").getLines()) {
      val series = loadSeriesFromAPI(seriesId)
      allSeries += (seriesId -> series)
      println(count + " of " + seriesCount + " imported")
      count = count + 1
    }
  }

  def loadSeriesFromAPI(seriesId: String): Seq[Observation] = {
    val response: HttpResponse[String] = Http("https://api.stlouisfed.org/fred/series/observations").param("series_id", seriesId).param("realtime_start", "1930-01-01").param("api_key", "0ed28d55d3e9655415b8e31652c8a952").asString
    val xml = scala.xml.XML.loadString(response.body)
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd")
    val observations = (xml \\ "observation").map { observation =>
      Observation(format.parse((observation \\ "@date").text), toDouble((observation \\ "@value").text))
    }
    observations
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
