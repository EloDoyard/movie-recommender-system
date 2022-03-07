package predict

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

  val measurements_glob_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, globalAvgPredictor(train))
  }))
  val timings_glob_avg = measurements_glob_avg.map(t => t._2)

  val measurements_user_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, userAvgPredictor(train))
  }))
  val timings_user_avg = measurements_user_avg.map(t => t._2)

  val measurements_item_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, itemAvgPredictor(train))
  }))
  val timings_item_avg = measurements_item_avg.map(t => t._2)

  val measurements_pred = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, formulaPredictor(train))
  }))
  val timings_pred = measurements_pred.map(t => t._2)

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
          "1.GlobalAvg" -> ujson.Num(globalAvg(train)), 
          "2.User1Avg" -> ujson.Num(userAvgPredictor(train)(1,0)),  
          "3.Item1Avg" -> ujson.Num(itemAvgPredictor(train)(0,1)),   
          "4.Item1AvgDev" -> ujson.Num(computeAllDevs(train, computeAllUsersAvg(train))(1)), 
          "5.PredUser1Item1" -> ujson.Num(formulaPredictor(train)(1,1))
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(MAE(test, globalAvgPredictor(train))), 
          "2.UserAvgMAE" -> ujson.Num(MAE(test, userAvgPredictor(train))),  
          "3.ItemAvgMAE" -> ujson.Num(MAE(test, itemAvgPredictor(train))),   
          "4.BaselineMAE" -> ujson.Num(MAE(test, formulaPredictor(train)))  
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_glob_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_glob_avg)) 
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_user_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_user_avg)) 
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_item_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_item_avg)) 
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_pred)), 
            "stddev (ms)" -> ujson.Num(std(timings_pred)) 
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