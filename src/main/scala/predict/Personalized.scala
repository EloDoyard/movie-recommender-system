package predict

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


class PersonalizedConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Personalized extends App {
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
  
  // Compute here

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
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "P.1" -> ujson.Obj(
          "1.PredUser1Item1" -> ujson.Num(predictor(train)(simOnes)(1,1)), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(MAE(test, predictor(train)(simOnes)))         // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          "1.AdjustedCosineUser1User2" -> ujson.Num(adjustedCosine(train)(1,2)), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(predictor(train)(adjustedCosine(train))(1,1)),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(MAE(test, predictor(train)(adjustedCosine(train)))) // MAE when using adjusted cosine similarity
        ),
        "P.3" -> ujson.Obj(
          "1.JaccardUser1User2" -> ujson.Num(0.0), // Similarity between user 1 and user 2 (jaccard similarity)
          "2.PredUser1Item1" -> ujson.Num(0.0),  // Prediction item 1 for user 1 (jaccard)
          "3.JaccardPersonalizedMAE" -> ujson.Num(0.0) // MAE when using jaccard similarity
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

  def MAE(data: Seq[Rating], predict: (Int, Int)=> Double): Double = {
    val acc = data.foldLeft((0.0,0)){
      (acc, x)=> {
        (acc._1+(predict(x.user, x.item)-x.rating).abs, acc._2+1)
      }
    }
    acc._1/acc._2
  }

  def simOnes = (user1: Int, user2: Int)=> 1.0

  def predictor(train: Seq[Rating])(sim: ((Int, Int)=> Double)): (Int, Int)=> Double = {
    // Compute necessary values for the predictions
    val ratingsByItems = ratingByItems(train)
    val usAvgs = computeAllUsersAvg(train)
    lazy val globalAvgValue = globalAvg(train)
    
    (user: Int, item: Int)=> {
      // user that rated this item 
      val users = ratingsByItems.getOrElse(item, Nil)
      // get average rating value for this user or the global average if the user didn't rate anything in the train set
      val avg = usAvgs.getOrElse(user, globalAvgValue)

      // get similarity with current user and the rating 
      val simVal = users.map(x=> (x.rating, sim(user, x.user)))

      val sumSim = simVal.foldLeft((0.0, 0.0)){
        (acc, x)=>{
          (acc._1 + x._1*x._2, acc._2 + x._2.abs)
        }
      }
      // compute the prediction
      val denomSum = if (sumSim._2!=0) sumSim._1/sumSim._2 else 0.0
      avg + denomSum* scale(avg+ denomSum, avg)
    }
  }

  def groupBy(data:Seq[Rating])(key: Rating => Int): Map[Int, Seq[Rating]] = 
    data.foldLeft(Map[Int, Seq[Rating]]()){
      // The accumulator is a map mapping the key (an int) to a pair (Double, Int) corresponding to the running sum of the value we want to compute and the number of computed value respectively 
      (acc, x)=>{
        // We access the value already stored or get 0 if no value was stored for this key
        val cur:Seq[Rating] = acc.getOrElse(key(x), Seq[Rating]())
        // Update of the map
        acc + (key(x) -> (x+:cur))
      }
    }

  def ratingByUsers(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.user)
  
  def ratingByItems(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.item)

  def adjustedCosine(train:Seq[Rating]): (Int, Int)=> Double = {
    val usAvg = computeAllUsersAvg(train)
    lazy val globalAvgValue = globalAvg(train)

    val dev = (x: Rating) => {
      val avg = usAvg.getOrElse(x.user, globalAvgValue)
      (x.rating-avg)/scale(x.rating, avg)
    }

    val mapped = train.map(x=>(x.user, dev(x)))

    val normByUsers = mapped.foldLeft(Map[Int,Double]()){(acc,x) =>
       val cur:Double = acc.getOrElse(x._1, 0.0)
        // Update of the map
        acc + (x._1 -> (cur+x._2*x._2))
    }

    val itemsByU = ratingByUsers(train)

    (user1: Int, user2: Int)=> {
      val ratings1 = itemsByU.getOrElse(user1, Nil)
      val ratings2 = itemsByU.getOrElse(user2, Nil)
      if(ratings1.length==0 || ratings2.length==0) 0.0
      else{
        val items1 = ratings1.map(_.item)
        
        val items2 = ratings2.map(_.item)
        val inter = items1.toSet.intersect(items2.toSet)

        val remaining2 = ratings2.foldLeft(Map[Int, Double]()){
          (acc, x)=>{
            if(inter.contains(x.item)){ acc + (x.item -> (dev(x)/normByUsers(user2)))}
            else acc
          }
        }

        ratings1.foldLeft(0.0){
          (acc, x)=>{
            if(inter.contains(x.item)){
              val norm = normByUsers.getOrElse(user1, 0.0)
              if(normByUsers==0) acc
              else acc + dev(x) / norm *remaining2(x.item)
            } else acc
          }
        }
      }
    }
  }
}
