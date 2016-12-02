package com.sibyl

import scala.collection.immutable.Seq
import scala.io.Source
import scala.xml.Node
import scalaj.http.{Http, HttpResponse}

/**
  * Created by Jesh on 11/27/16.
  */
object FREDDataLoad {
  case class Observation(date: String, value: String)

  def main(args: Array[String]): Unit = {
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
    val response: HttpResponse[String] = Http("https://api.stlouisfed.org/fred/series/observations").param("series_id", seriesId).param("api_key", "0ed28d55d3e9655415b8e31652c8a952").asString
    val xml = scala.xml.XML.loadString(response.body)
    val data = (xml \\ "observations")
    val observations = (xml \\ "observation").map { observation =>
      Observation((observation \\ "@date").text, (observation \\ "@value").text)
    }
    return observations
  }

  def handleEmptyIntNode(node: Option[Seq[Node]], default: Int = 0): Int = {
    if(node.isEmpty) default
    else if(node.get.isEmpty) default
    else node.get(0).text.toInt
  }

  def handleEmptyStringNode(node: Option[Seq[Node]], default: String = ""): String = {
    if(node.isEmpty) default
    else if(node.get.isEmpty) default
    else node.get(0).text
  }
}
