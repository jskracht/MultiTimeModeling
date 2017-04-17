package com.sibyl

import java.io.File
import java.sql.Timestamp
import java.time.{LocalDateTime, ZoneId, ZonedDateTime}

import org.apache.spark.sql.SparkSession

class CSVDataLoad(val spark: SparkSession) {

  //Convert FRED YYYY-MM-DD formated string to date
  private def convertStringToDate(date: String): Timestamp = {
    Timestamp.from(ZonedDateTime.of(LocalDateTime.parse(date + "T00:00:00"), ZoneId.systemDefault()).toInstant)
  }

  //Convert FRED value column to double
  private def parseDoubleOrZero(value: String): Double = {
    try {
      value.toDouble
    } catch {
      case _: Exception => null.asInstanceOf[Double]
    }
  }

  //Read in CSVs
  def loadCSVData(): Unit = {
    for (file <- listFiles(new File("data/FRED")).filter(_.getName.contains("csv"))) {
      println(file.getPath)
      spark.read.format("csv").option("header", "true").load(file.getPath)
    }
  }

  final def listFiles(base: File, recursive: Boolean = true): Seq[File] = {
    val files = base.listFiles
    val result = files.filter(_.isFile)
    result ++
      files
        .filter(_.isDirectory)
        .flatMap(listFiles(_, recursive))
  }
}