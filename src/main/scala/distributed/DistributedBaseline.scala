package distributed

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._
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
    //MeanAbsoluteErrorSpark(baselinePredictorSpark(train,test), test)
    Thread.sleep(1000) // Do everything here from train and test
    42 
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
          "1.GlobalAvg" -> ujson.Num(0.0),//getGlobalAvg(train)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(0.0),//getUsersAvg(train)(1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(0.0),//getUsersAvg(train)(1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(0.0),//getItemsAvgDev(train)(1)), // Datatype of answer: Double,
          "5.PredUser1Item1" -> ujson.Num(0.0),//baselinePredictorSpark(train,test)(1,1)), // Datatype of answer: Double
          "6.Mae" -> ujson.Num(0.0)//MeanAbsoluteErrorSpark(baselinePredictorSpark(train, test),test)) // Datatype of answer: Double
        ),
        "D.2" -> ujson.Obj(
          "1.DistributedBaseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)), 
            "stddev (ms)" -> ujson.Num(std(timings)) 
          )            
        )
      )
      val json = write(answers, 4)

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }  

//   def meanSpark(ratings : RDD[Double]) : Double = {
//     ratings.sum/ratings.count()
//   }

//   def MeanAbsoluteErrorSpark(predictor : (Int, Int) => Double, real : RDD[Rating]) : Double={
//     meanSpark(real.map(x=> (predictor(x.user, x.item)-x.rating).abs))
//   }

//   def getGlobalAvg(ratings: RDD[Rating]) : Double = meanSpark(ratings.map(_.rating))

//   def getUsersAvg(ratings : RDD[Rating]) : collection.Map[Int,Double] = ratings.map{
//     case x : Rating => x.user->(1,x.rating)
//     }.reduceByKey((acc, a) => (acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collectAsMap

//   def getItemsAvg(ratings:RDD[Rating]) : collection.Map[Int,Double] = ratings.map{
//     case x : Rating => x.item->(1,x.rating)
//   }.reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collectAsMap
//   // .groupBy(_.item).mapValues(x=>meanSpark(x.map(_.rating)))
  
//   def getItemsAvgDev (ratings : RDD[Rating]) : collection.Map[Int,Double] = {
//     val usersAvg = getUsersAvg(ratings)
//     val itemsAvg = getItemsAvg(ratings)
//     val globalAvg = getGlobalAvg(ratings)
//     val deviation = ratings.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg))))
//     deviation.map{
//       case x : Rating => x.item->(1,x.rating)
//     }.reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collectAsMap
//  }

//   def baselinePredictorSpark (ratings : RDD[Rating], to_pred:RDD[Rating]) : (Int, Int) => Double = {
//     val globalAvg = getGlobalAvg(ratings)
//     val usersAvg = getUsersAvg(ratings)
//     val itemsAvg = getItemsAvg(ratings)
//     val deviationAvg = getItemsAvgDev(ratings)
//     // potentiellement faut de faire map sur to_pred
//     // val deviationPred = ratings.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user))/scale(x.rating, usersAvg.getOrElse(x.user)))) 
//     val deviationPred = to_pred.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg)))) 
//     var devAvgPred = deviationPred.map{
//       case x : Rating => x.item->(1,x.rating)
//     }.reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collectAsMap
//     (u,i) => usersAvg.getOrElse(u, globalAvg)+devAvgPred(i)*scale(usersAvg.getOrElse(u, globalAvg)+devAvgPred(i), usersAvg.getOrElse(u, globalAvg))
//   }

  println("")
  spark.close()
}