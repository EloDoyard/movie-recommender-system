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
          "4.PredUser1Item1" -> ujson.Num(predictKNN(train, 10)(1,1))//Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> List(10,30,50,100,200,300,400,800,943).map(k => 
              List(
                k,
                0.0//MeanAbsoluteError(predictKNN(train, k), test) // Compute MAE
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

  println(predictKNN(train, 50)(1,1))
  println(predictor(train)(getSimilarity(train, 50, adjustedCosineSimilarityFunction(train)))(1,1))
  // println(predictKNN(train, 50)(4,5))
  // println(predictKNN(train, 50)(3,200))
  // println(predictKNN(train, 50)(7,10))
  // println(predictKNN(train, 50)(8,11))
  // println(predictKNN(train, 50)(50,712))

  def getNeighbors(ratings : Seq[Rating], k : Int) : Int => Seq[(Int, Double)] = {
    // ratings.map(_.user).toSet.toList.permutations(2).groupBy(x._1).mapValues(
    //   y=>y.map(
    //     x=> (x, adjustedCosineSimilarityFunction(x._1, x._2))).toSeq.sortWith(_._2>_._2).take(k))

    val similarityFunction = adjustedCosineSimilarityFunction(ratings)

    val allUsers = ratings.map(x=>x.user).toSet
    var allNeighbors = Map[Int, Seq[(Int, Double)]]()

    u => {
      var uNeighbors = allNeighbors.getOrElse(u, Nil)
      if (uNeighbors.isEmpty) {
        val others = (allUsers- u).toSeq
        uNeighbors = others.map(x=>(x, similarityFunction(u,x))).sortBy(_._2)(Ordering[Double].reverse).take(k)
        allNeighbors = allNeighbors+(u-> uNeighbors)
      }
      uNeighbors
      // (allUsers-u).toSeq.map(x=>(x, similarityFunction(u,x))).sortWith(_._2>_._2).take(k)
    }

    // val neighbors = ratings.map(_.user).toSet.toList.combinations(2).map{ case List(a, b) => (a, b) }.toSeq.foldLeft(
    //   Map[Int, Seq[((Int, Int), Double)]]()){
    //   (acc, a) => {
    //     acc+(
    //       a._1 -> (((a, similarityFunction(a._1, a._2))) +: acc.getOrElse(a._1, Seq())), 
    //       a._2 -> ((((a._2,a._1), similarityFunction(a._1,a._2)))+:acc.getOrElse(a._2, Seq())))
    //   }
    // }

    // // neighbors.mapValues(x=> x._.sortWith(_._2>_._2).take(k))
    // neighbors.foldLeft(Map[Int, Seq[((Int, Int), Double)]]()) {
    //   (acc, a) => {
    //     acc+(a._1 -> a._2.sortWith(_._2>_._2).take(k))
    //   }
    // }
    // neighbors.transform{(key, value) => value.sortWith(_._2>_._2).take(k)}
  } 

  def getSimilarity (ratings : Seq[Rating], k : Int, sim: (Int, Int)=> Double) : (Int, Int) => Double = {
    val nn = kNearestNeighbours(ratings, k, sim)
    (user1: Int, user2: Int) => {
      nn(user1).map(x=>{
        if (x._1==user2){x._2}
        else {0.0}
        }).sum
    }
  }

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

  def weightedSumDevKNN (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {

    val globalAvgValue = globalAvg(ratings)
    // find k-nearest neighbors per user
    val allNeighbors = getNeighbors(ratings, k)
    
    // val normalizedDeviations = computeNormalizeDeviation(ratings)
    val simFunction = getSimilarity(ratings, k, adjustedCosineSimilarityFunction(ratings))
    //var ratedI = computeRatedI(ratings)
    val usersAvgValue = usersAvg(ratings)

    var ratingsByItems = ratings.groupBy(_.item)

    // val usersItem = ratings.groupBy(_.item)

    (u : Int,i : Int) => {
      // val usersTemp = ratingsByItems.getOrElse(item, Array[Rating]())
      val users = ratingsByItems.getOrElse(i, Seq[Rating]())
      // println(users)

      val userAvg = usersAvgValue.getOrElse(u, globalAvgValue)

      val neighbors = allNeighbors(u).toMap

      // users that rated this item
      // println(users)
      // val simVal = temp.map(x=> {
      //   val xAvg = usersAvgValue.getOrElse(x.user, globalAvgValue)
      //   ((x.rating-xAvg)/scale(x.rating, xAvg), simFunction(x.user, u))
      // })

      val simVal = users.map(x=> {
        val avgU = usersAvgValue.getOrElse(x.user, globalAvgValue)
        ((x.rating-avgU)/scale(x.rating, avgU), simFunction(u, x.user))
      })

      val ssSum = simVal.foldLeft((0.0,0.0)) {
        (acc, a) => {
          (acc._1+a._1*a._2, acc._2+a._2.abs)
        }
      }
      // println(neighbors)
      // val ssSum = neighbors.map(_._2.abs).sum
      // println(ssSum)
      // println(neighbors)
      if (ssSum._2 >0) {
        ssSum._1/ssSum._2
        // neighbors.map(x=> x._2*normalizedDeviations.getOrElse((x._1, i), 0.0)).sum/ssSum
      } else 0.0
      
    }
    }
    // // Compute mandatory values for the predictions
    // val ratingsByItems = train.groupBy(_.item)
    // val usAvgs = usersAvg(train)
    // val globalAvgValue = globalAvg(train)

    // val sim = getSimilarity(ratings, k, adjustedCosineSimilarityFunction(train))
   
    // // val sim = getadjustedCosineSimilarityFunction(train)
    
    // (user: Int, item: Int)=> {
    //   // user that rated this item
    //   val users = ratingsByItems.getOrElse(item, Array[Rating]())
    //   print(users.getClass)

    //   // get average rating value for this user or the global average if the user didn't rate anything in the train set
    //   val avg = usAvgs.getOrElse(user, globalAvgValue)

    //   // get similarity with current user and the rating 
    //   val simVal = users.map(x=> {
    //     val avgU = usAvgs.getOrElse(x.user, globalAvgValue)
    //     ((x.rating-avgU)/scale(x.rating, avgU), sim(user, x.user))
    //   })

    //   // Compute the denominator as well as the numerator of the similarity between the two useres 
    //   val sumSim = simVal.foldLeft((0.0, 0.0)){
    //     (acc, x)=>{
    //       (acc._1 + x._1*x._2, acc._2 + x._2.abs)
    //     }
    //   }
    //   // compute the prediction
    //   if (sumSim._2!=0) sumSim._1/sumSim._2 else 0.0
    // }
  // }

  def predictor(train: Seq[Rating]) (sim: ((Int, Int)=> Double)) : (Int, Int)=> Double = {
    // Compute mandatory values for the predictions
    val ratingsByItems = train.groupBy(_.item)
    val usAvgs = usersAvg(train)
    val globalAvgValue = globalAvg(train)

    // val sim = getadjustedCosineSimilarityFunction(train)
    
    (user: Int, item: Int)=> {
      // user that rated this item 
      val users = ratingsByItems.getOrElse(item, Nil)

      // get average rating value for this user or the global average if the user didn't rate anything in the train set
      val avg = usAvgs.getOrElse(user, globalAvgValue)

      // get similarity with current user and the rating 
      val simVal = users.map(x=> {
        val avgU = usAvgs.getOrElse(x.user, globalAvgValue)
        ((x.rating-avgU)/scale(x.rating, avgU), sim(user, x.user))
      })

      // Compute the denominator as well as the numerator of the similarity between the two useres 
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


  def predictKNN(ratings : Seq[Rating], k : Int) : (Int, Int)=> Double = {
    val globalAvgValue = globalAvg(ratings)
    val usersAvgValue = usersAvg(ratings)

    val wsdKNN = weightedSumDevKNN(ratings, k)
    (u,i) => {
      var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
      var userWSD = wsdKNN(u, i)
      userAvg+userWSD*scale(userAvg+userWSD, userAvg)
    }
  }

  println("")
  spark.close()
}
