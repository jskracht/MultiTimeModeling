package com.sibyl

import java.sql.{Connection, DriverManager}
import org.apache.spark.sql.SparkSession

class JDBCConnection(val spark: SparkSession) {

  //Connect to Database
  def getJDBCConnection: Connection = {
    val jdbcUsername = "Jesh"
    val jdbcPassword = "Jk-yosmite1"
    val jdbcHostname = "localhost"
    val jdbcPort = 5432
    val jdbcDatabase ="postgres"
    val jdbcUrl = s"jdbc:postgresql://$jdbcHostname:$jdbcPort/$jdbcDatabase?user=$jdbcUsername&password=$jdbcPassword"
    val connectionProperties = new java.util.Properties()
    Class.forName("org.postgresql.Driver")
    DriverManager.getConnection(jdbcUrl, jdbcUsername, jdbcPassword)
  }
}