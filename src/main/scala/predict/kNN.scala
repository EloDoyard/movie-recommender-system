package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._
import predict.Baseline._
import predict.Personalized._

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
    //MeanAbsoluteError(predictKNN(train, 300),test)
    Thread.sleep(1000)
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
          "3.Measurements" -> conf.num_measurements()
        ),
        "N.1" -> ujson.Obj(
          "1.k10u1v1" -> ujson.Num(getSimilarity(train, 10, adjustedCosineSimilarityFunction(train))(1,1)), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(getSimilarity(train, 10, adjustedCosineSimilarityFunction(train))(1,864)), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(getSimilarity(train, 10, adjustedCosineSimilarityFunction(train))(1,886)), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(predictKNN(train, 10)(1,1)) //Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> List(10,30,50,100,200,300,400,800,943).map(k => 
              List(
                k,
                MeanAbsoluteError(predictKNN(train, k), test) // Compute MAE
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

 /**
  * Compute all the neighbors of a user and their similarities
  * @param ratings a sequence of ratings
  * @param k number of neighbors to keep
  * @return function that map user to sequence of neighbors and their similarity score
  */
  def getNeighbors(ratings : Seq[Rating], k : Int) : Int => Seq[(Int, Double)] = {
  
    // defined adjusted cosine similairy function 
    val similarityFunction = adjustedCosineSimilarityFunction(ratings)

    // set of users in ratings dataset
    val allUsers = ratings.map(x=>x.user).toSet
    // initiation of neighbors map
    var allNeighbors = Map[Int, Seq[(Int, Double)]]()

    u => {
      // get all neighbors of u if already computed
      var uNeighbors = allNeighbors.getOrElse(u, Nil)
      if (uNeighbors.isEmpty) { // neighbors of u never computed
        // all possible neighbors of u
        val others = (allUsers- u).toSeq
        // compute similarity between u and possible neighbors and choose k most similar
        uNeighbors = others.map(x=>(x, similarityFunction(u,x))).sortWith(_._2>_._2).take(k)
        // update neighbors map
        allNeighbors = allNeighbors+(u-> uNeighbors)
      }
      // u's neighbors along with their similarity score
      uNeighbors
    }
  } 

  /**
  * Compute the similarity of pair of users taking into consideration neighborhood
  * @param ratings a sequence of ratings
  * @param k number of neighbors to consider
  * @param sim similairy function to user
  * @return function that map (user1, user2) to their similarity score
  */
  def getSimilarity (ratings : Seq[Rating], k : Int, sim: (Int, Int)=> Double) : (Int, Int) => Double = {
    // map each user to its neighborhood
    val nn = getNeighbors(ratings, k)
    // compute similarity between 2 users
    (user1: Int, user2: Int) => {
      nn(user1).map(x=>{
        if (x._1==user2){x._2}
        else {0.0}
        }).sum
    }
  }

  /**
  * Compute user-specific weighted-sum deviation function taking into consideration its neighborhood
  * @param ratings a sequence of ratings
  * @param k number of nieghbors to keep in neighborhood
  * @return map each (user, item) to its weighted-sum deviation
  */
  def weightedSumDevKNN (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {

    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // find k-nearest neighbors per user
    val allNeighbors = getNeighbors(ratings, k)
    
    // define similarity function
    val simFunction = getSimilarity(ratings, k, adjustedCosineSimilarityFunction(ratings))
    // map each user to its average rating
    val usersAvgValue = usersAvg(ratings)

    // map each item to sequence of ratings
    var ratingsByItems = ratings.groupBy(_.item)

    (u : Int,i : Int) => {
      // sequence of ratings that rated i
      val users = ratingsByItems.getOrElse(i, Seq[Rating]())

      // user u average rating
      val userAvg = usersAvgValue.getOrElse(u, globalAvgValue)

      // u's neighborhood
      val neighbors = allNeighbors(u).toMap

      // compute for each ratings that rated i the similarity score with u and the normalized deviation
      val simVal = users.map(x=> {
        // get x's user average rating
        val avgU = usersAvgValue.getOrElse(x.user, globalAvgValue)
        ((x.rating-avgU)/scale(x.rating, avgU), simFunction(u, x.user))
      })

      // compute numerator and denominator of weighted-sum deviation
      val ssSum = simVal.foldLeft((0.0,0.0)) {
        (acc, a) => {
          (acc._1+a._1*a._2, acc._2+a._2.abs)
        }
      }

      // return user u weighted-sum deviation
      if (ssSum._2 >0) {
        ssSum._1/ssSum._2
      } else 0.0
      
    }
    }


  /**
  * kNN Predictor predicting for each (user, item) its baseline prediction taking into consideration the neighborhood
  *   and the user-specific weighted-sum deviation
  * @param ratings a sequence of ratings
  * @param k number of neighbors to consider in neighborhood
  * @return map item to its average deviation
  */
  def predictKNN(ratings : Seq[Rating], k : Int) : (Int, Int)=> Double = {
    // compute global average rating
    val globalAvgValue = globalAvg(ratings)
    // map each user to its average rating
    val usersAvgValue = usersAvg(ratings)
    // map each (user, item) to its weighted-sum deviation
    val wsdKNN = weightedSumDevKNN(ratings, k)

    (u,i) => {
      // get u's average rating
      var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
      // get (u,i)'s deviation
      var userWSD = wsdKNN(u, i)
      // compute prediction
      userAvg+userWSD*scale(userAvg+userWSD, userAvg)
    }
  }

  println("")
  spark.close()
}
