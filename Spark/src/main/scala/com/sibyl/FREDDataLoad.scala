package com.sibyl

import java.sql.Timestamp
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}
import java.util.Date

import com.cloudera.sparkts.{DateTimeIndex, DayFrequency, TimeSeriesRDD}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.udf

import scala.io.Source
import scala.util.parsing.json._
import org.apache.spark.sql.types._

object FREDDataLoad extends App {
  case class Point(timestamp: Timestamp, value: Double)
  case class Series(id: String, points: List[Point])
  val spark = SparkSession.builder.master("local").appName("Sibyl").getOrCreate()

  //Get HTTP Response from URL
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
  val seriesIDs = Source.fromFile("data/seriesList2").getLines().toList
  for (series <- seriesIDs) {
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

  //Create DataFrame
  val rowRDD = spark.sparkContext.parallelize(rows)
  val fields = Seq(
    StructField("date", TimestampType, true),
    StructField("series", StringType, true),
    StructField("rawValue", DoubleType, true)
  )
  val schema = StructType(fields)
  val rawTimeSeriesData = spark.sqlContext.createDataFrame(rowRDD, schema)

  //Vectorize raw values to use Min/Max Scaler for normalization
  val vectorize = udf((rawValue:Double) => Vectors.dense(Array(rawValue)))
  val vectorizedTimeSeriesData = rawTimeSeriesData.withColumn("rawValue", vectorize(rawTimeSeriesData("rawValue")))
  val scaler = new MinMaxScaler()
    .setInputCol("rawValue")
    .setOutputCol("value")
    .setMax(1)
    .setMin(-1)

  val normalizedTimeSeriesData = scaler.fit(vectorizedTimeSeriesData).transform(vectorizedTimeSeriesData)

  //Convert vectorized values back to doubles
  val devectorize = udf{ value:DenseVector => value(0) }
  val devectorizedNormalizedTimeSeriesData = normalizedTimeSeriesData.withColumn("value", devectorize(normalizedTimeSeriesData("value")))
  devectorizedNormalizedTimeSeriesData.show()

  //Create DateTimeIndex
  val zone = ZoneId.systemDefault()
  val dateTimeIndex = DateTimeIndex.uniformFromInterval(
    ZonedDateTime.of(LocalDateTime.parse("1960-01-01T00:00:00"), zone),
    ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone), new DayFrequency(1))

  //Put data into TimeSeriesRDD
  val timeSeriesRDD = TimeSeriesRDD.timeSeriesRDDFromObservations(dateTimeIndex, devectorizedNormalizedTimeSeriesData,
    "date", "series", "value")

  //Cache in memory
  timeSeriesRDD.cache()

  //Fill in null values using linear interpolation
  val filledTimeSeriesRDD = timeSeriesRDD.fill("linear")
  val sliced = filledTimeSeriesRDD.slice(ZonedDateTime.of(LocalDateTime.parse("1996-01-01T00:00:00"), zone), ZonedDateTime.of(LocalDateTime.parse("2017-01-01T00:00:00"), zone))

  println(sliced.collect().head)
}