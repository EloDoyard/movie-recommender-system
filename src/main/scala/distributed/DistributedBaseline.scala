package distributed

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._

// custom import 
import predict.Baseline._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val master = opt[String](default=Some(""))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object DistributedBaseline extends App {
  var conf = new Conf(args) 

  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = if (conf.master() != "") {
    SparkSession.builder().master(conf.master()).getOrCreate()
  } else {
    SparkSession.builder().getOrCreate()
  }
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator())
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator())

  val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, predictorFunctionSpark(train))
  }))
  val timings = measurements.map(t => t._2) // Retrieve the timing measurements

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
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> conf.train(),
          "2.Test" -> conf.test(),
          "3.Master" -> conf.master(),
          "4.Measurements" -> conf.num_measurements()
        ),
        "D.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(predictorGlobalAvgSpark(train)(1,0)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(predictorUserAvgSpark(train)(1,0)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(predictorItemAvgSpark(train)(1,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(computeItemDevsSpark(train,computeAllUsersAvgSpark(train))(1)), // Datatype of answer: Double,
          "5.PredUser1Item1" -> ujson.Num(predictorFunctionSpark(train)(1,1)), // Datatype of answer: Double
          "6.Mae" -> ujson.Num(MAE(test, predictorFunctionSpark(train))) // Datatype of answer: Double
        ),
        "D.2" -> ujson.Obj(
          "1.DistributedBaseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings)) // Datatype of answer: Double
          )            
        )
      )
      val json = write(answers, 4)

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
    

  }

  println("")
  spark.close()

  def applyAndAverage(data: RDD[Rating])(f: (Rating => Double)): Double = {
    val acc = data.map(x => (f(x), 1)).reduce( (x,y) => (x._1 + y._1, x._2 + y._2))
    acc._1/acc._2
  }

  def predictorFunctionSpark(data: RDD[Rating]):(Int, Int )=> Double = {
    val usersAvg = computeAllUsersAvgSpark(data)
    val globalAvgValue = computeGlobalAvgSpark(data)
    val devs = computeItemDevsSpark(data, usersAvg)
    (user, item)=>{
      val dev = devs.getOrElse(item, 0.0)
      val avg = usersAvg.getOrElse(user, globalAvgValue)
      avg + dev*scale(dev + avg, avg)
    }
  }

  def predictorUserAvgSpark(data: RDD[Rating]): (Int, Int )=> Double = {
    val usersAvg = computeAllUsersAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => usersAvg.getOrElse(user, globalAvgValue)
  }

  def predictorItemAvgSpark(data: RDD[Rating]): (Int, Int)=> Double = {
    val itemsAvg = computeItemAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => itemsAvg.getOrElse(item, globalAvgValue)
  }

  def predictorGlobalAvgSpark(data: RDD[Rating]): (Int, Int) => Double = {
    val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => globalAvgValue
  }

  def MAE(test: RDD[Rating], predict: (Int, Int)=> Double): Double = applyAndAverage(test){x=> (x.rating-predict(x.user, x.item)).abs}

  def computeGlobalAvgSpark(data: RDD[Rating]): Double = applyAndAverage(data)(_.rating)

  def computeAllUsersAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.user).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap

  def computeItemAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap

  def computeItemDevsSpark(data: RDD[Rating], usAvg: Map[Int, Double]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq){x=>
      val avg = usAvg(x.user)
      (x.rating-avg)/scale(x.rating, avg)
    }
  }).collect().toMap
}