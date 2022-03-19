package recommend

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import scala.util.Sorting
import predict.kNN._
import predict.Baseline._
import predict.Personalized._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val data = opt[String](required = true)
  val personal = opt[String](required = true)
  val separator = opt[String](default = Some("\t"))
  val json = opt[String]()
  verify()
}

object Recommender extends App {
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
  println("Loading data from: " + conf.data()) 
  val data = load(spark, conf.data(), conf.separator()).collect()
  assert(data.length == 100000, "Invalid data")

  println("Loading personal data from: " + conf.personal()) 
  val personalFile = spark.sparkContext.textFile(conf.personal())
  val personal = personalFile.map(l => {
      val cols = l.split(",").map(_.trim)
      if (cols(0) == "id") 
        Rating(944,0,0.0)
      else 
        if (cols.length < 3) 
          Rating(944, cols(0).toInt, 0.0)
        else
          Rating(944, cols(0).toInt, cols(2).toDouble)
  }).filter(r => r.rating != 0).collect()
  val movieNames = personalFile.map(l => {
      val cols = l.split(",").map(_.trim)
      if (cols(0) == "id") (0, "header")
      else (cols(0).toInt, cols(1).toString)
  }).collect().toMap


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
      val augmented = data.union(personal)
      // println(augmented.size)
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "data" -> conf.data(),
          "personal" -> conf.personal()
        ),
        "R.1" -> ujson.Obj(
          "PredUser1Item1" -> ujson.Num(predictKNN(augmented, 300)(1,1)) //0.0) // Prediction for user 1 of item 1
        ),
          // IMPORTANT: To break ties and ensure reproducibility of results,
          // please report the top-3 recommendations that have the smallest
          // movie identifier.

        "R.2" -> recommendations(augmented, 300)(944, 3).map(x => ujson.Arr(x._1, movieNames(x._1), x._2))
       )
      val json = write(answers, 4)

      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }

  def recommendations (ratings : Seq[Rating], k : Int) : (Int, Int) => Seq[(Int, Double)] = {
    val knn = predictKNN(ratings, k)
    // println(knn.getClass)

    val order = (x:(Int, Double), y:(Int, Double)) => {
      if (x._2 == y._2) {
        x._1<y._1
      } else {
        x._2>y._2
      }
    }
    (user : Int, n : Int) => {
      val notRated = ratings.map(_.item).toSet.diff(ratings.filter(x=> x.user == user).map(_.item).toSet)
      val predictions = notRated.toSeq.map(x=> (x, knn(user,x)))

      predictions.sortWith(order).take(n)
    }
    

  }


  println("")
  spark.close()
}
