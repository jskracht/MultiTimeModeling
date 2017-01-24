package com.sibyl

import java.sql.Date

import scala.io.Source
import scala.util.parsing.json._

object FREDDataLoad2 extends App {

  @throws(classOf[java.io.IOException])
  @throws(classOf[java.net.SocketTimeoutException])
  def get(url: String,
          connectTimeout:Int =5000,
          readTimeout:Int =5000,
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

  def parseDoubleOrZero(s : String): Double = try { s.toDouble }  catch { case _ :Exception => null.asInstanceOf[Double] }

  val seriesIds = Source.fromFile("data/seriesList").getLines().toList
  for (series <- Source.fromFile("data/seriesList").getLines()) {
    println("Fetching Series: " + series)

    val content = get("https://api.stlouisfed.org/fred/series/observations?series_id=%s&api_key=%s&file_type=json".
      format(series, "0ed28d55d3e9655415b8e31652c8a952")
    )
    val respFromFRED = JSON.parseFull(content)
    val observations = respFromFRED.get.asInstanceOf[Map[String, Any]]("observations").asInstanceOf[List[Any]]

    for(singleOb <- observations) {
      val insertStr = "%s, %g".
        format(
          singleOb.asInstanceOf[Map[String, Date]]("date").asInstanceOf[String],
          parseDoubleOrZero(singleOb.asInstanceOf[Map[String, Double]]("value").asInstanceOf[String])
        )


      println(insertStr)
    }
  }
}