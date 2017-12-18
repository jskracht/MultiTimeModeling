name := "New"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.1" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.1"
libraryDependencies += "org.apache.spark" %% "spark-streaming-flume" % "2.2.1"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.4"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.4" % "test"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.7.0"
libraryDependencies += "org.scalaj" %% "scalaj-http" % "2.3.0"
libraryDependencies += "com.cloudera.sparkts" % "sparkts" % "0.4.1"
libraryDependencies += "org.postgresql" % "postgresql" % "42.1.4"