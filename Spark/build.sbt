name := "Spark"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.2" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.2"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.0.2"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.0.2"
libraryDependencies += "org.apache.spark" %% "spark-streaming-flume" % "2.0.2"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"
libraryDependencies += "org.scalaj" %% "scalaj-http" % "2.3.0"