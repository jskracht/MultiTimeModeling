package com.sibyl

import java.sql.Timestamp
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}
import java.util.Date

import com.cloudera.sparkts.{BusinessDayFrequency, DateTimeIndex, TimeSeriesRDD}
import org.apache.spark.sql.{Row, SparkSession}

import scala.io.Source
import scala.util.parsing.json._
import org.apache.spark.sql.types._

object FREDDataLoad extends App {
  case class Point(timestamp: Timestamp, value: Double)
  case class Series(id: String, points: List[Point])
  val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()

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

  def convertStringToDate(date: String): Timestamp = Timestamp.from(ZonedDateTime.of(LocalDateTime.parse(date + "T00:00:00"), ZoneId.systemDefault()).toInstant)

  def parseDoubleOrZero(value : String): Double = try { value.toDouble }  catch { case _ :Exception => null.asInstanceOf[Double] }

  var rows = scala.collection.mutable.ListBuffer[Row]()
  var parallelLists = scala.collection.mutable.Buffer[Series]()
  val seriesIds = Source.fromFile("data/seriesList2").getLines().toList
  for (series <- seriesIds) {
    println("Fetching Series: " + series)

    val content = get("https://api.stlouisfed.org/fred/series/observations?series_id=%s&api_key=%s&file_type=json".
      format(series, "0ed28d55d3e9655415b8e31652c8a952")
    )
    val respFromFRED = JSON.parseFull(content)
    val observations = respFromFRED.get.asInstanceOf[Map[String, Any]]("observations").asInstanceOf[List[Any]]


    for(singleOb <- observations) {
      val date = convertStringToDate(singleOb.asInstanceOf[Map[String, Date]]("date").asInstanceOf[String])
      val value = parseDoubleOrZero(singleOb.asInstanceOf[Map[String, Double]]("value").asInstanceOf[String])
      rows += Row(date, series, value)
    }
  }

  val rowRdd = spark.sparkContext.parallelize(rows)
  val fields = Seq(
    StructField("date", TimestampType, true),
    StructField("series", StringType, true),
    StructField("value", DoubleType, true)
  )
  val schema = StructType(fields)
  val data = spark.sqlContext.createDataFrame(rowRdd, schema)
  data.show()

  val zone = ZoneId.systemDefault()
  val dtIndex = DateTimeIndex.uniformFromInterval(
    ZonedDateTime.of(LocalDateTime.parse("1996-01-01T00:00:00"), zone),
    ZonedDateTime.of(LocalDateTime.parse("1999-01-01T00:00:00"), zone),
    new BusinessDayFrequency(1))

  val tickerTsrdd = TimeSeriesRDD.timeSeriesRDDFromObservations(dtIndex, data,
    "date", "series", "value")

  tickerTsrdd.cache()
  val filled = tickerTsrdd.fill("linear")
}