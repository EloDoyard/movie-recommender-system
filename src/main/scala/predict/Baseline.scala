package predict
//package co.kbhr.scaladoc_tags

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Baseline extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new Conf(args) 
  // For these questions, data is collected in a scala Array 
  // to not depend on Spark
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()

  // global average
  val globalMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(computeAvgRating(train), test)
    
  }))
  val globalTime = globalMeasure.map(t => t._2) // Retrieve the timing measurements

  // user average
  val userMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(computeUserAvg(train), test)
    
  }))
  val userTime = userMeasure.map(t => t._2) // Retrieve the timing measurements

  // item average
  val itemMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(computeItemAvg(train), test)
    
  }))
  val itemTime = itemMeasure.map(t => t._2) // Retrieve the timing measurements

  // baseline
  val baselineMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(computePrediction(train), test)
    
  }))
  val baselineTime = baselineMeasure.map(t => t._2) // Retrieve the timing measurements

  // Save answers as JSON
  def printToFile(content: String, 
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }

  conf.json.toOption match {
    case None => ; 
    case Some(jsonFile) => {


      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(computeAvgRating(train)(1,1)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(computeUserAvg(train)(1,1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(computeItemAvg(train)(1,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(computeItemAvgDev(train)(1,1)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(computePrediction(train)(1,1)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(MAE(computeAvgRating(train), test)), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(MAE(computeUserAvg(train), test)),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(MAE(computeItemAvg(train), test)),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(MAE(computePrediction(train), test))   // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(globalTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(globalTime)) // Datatype of answer: Double

          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(userTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(userTime)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(itemTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(itemTime)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(baselineTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(baselineTime)) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  println("")
  spark.close()
}