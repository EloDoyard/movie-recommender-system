package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class kNNConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object kNN extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new PersonalizedConf(args) 
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()


  val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MAE(test, predictor(train)(getSimilarity(train, 300, adjustedCosine(train))))
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
          "3.Measurements" -> conf.num_measurements()
        ),
        "N.1" -> ujson.Obj(
          "1.k10u1v1" -> ujson.Num(getSimilarity(train, 10, adjustedCosine(train))(1,1)), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(getSimilarity(train, 10, adjustedCosine(train))(1,864)), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(getSimilarity(train, 10, adjustedCosine(train))(1,886)), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(MAE(test, predictor(train)(getSimilarity(train, 10, adjustedCosine(train))))) // Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> List(10,30,50,100,200,300,400,800,943).map(k => 
              List(
                k,
                MAE(test, predictor(train)(getSimilarity(train, k, adjustedCosine(train))))
              )
          ).toList
        ),
        "N.3" -> ujson.Obj(
          "1.kNN" -> ujson.Obj(
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

  println("")
  spark.close()

  /**
  * Get the similarity of two users based on k-nearest neighbours
  * @param train the training set on which to fit the predictor
  * @param k the number of neighbors we want to take into account
  * @param sim function measuring the similarity between two users
  * @return a function returning the similarity between two users in the knn sense
  */
  def getSimilarity(train: Seq[Rating], k: Int, sim: (Int, Int)=> Double): (Int, Int)=> Double = {
    val nn = kNearestNeighbours(train, k, sim)
    (user1: Int, user2: Int) => {
      nn(user1).map(x=>{
        if (x._1==user2){x._2}
        else {0.0}
        }).sum
    }
  }

  /**
  * Return the list of the k nearest neighbours as well as their similarity for the given user
  * @param train the training set on which to fit the predictor
  * @param k the number of neighbors we want to take into account
  * @param sim function measuring the similarity between two users
  * @return function mapping a user id to the list of k (neigbourdid, similarity) pairs 
  */
  def kNearestNeighbours(train: Seq[Rating], k: Int,  sim: (Int, Int)=> Double): Int => List[(Int, Double)] = {
    val allUsers = train.map(x=>x.user).toSet
    var map = Map[Int, List[(Int, Double)]]() // map that will act as a local cache
    (user: Int )=>{
      var neighbours = map.getOrElse(user, Nil)
      if (neighbours.length == 0){
        val allOthers = (allUsers-user).toList
        neighbours = allOthers.map(x=> (x, sim(user, x))).sortBy(_._2)(Ordering[Double].reverse).take(k)
        map = map +(user-> neighbours)
      }
      neighbours
    }
  }
}
