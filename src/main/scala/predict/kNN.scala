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
          "1.k10u1v1" -> ujson.Num(0.0),//getSimilarity(train, 10)(1,1)), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(0.0),//getSimilarity(train, 10)(1,864)), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(0.0),//getSimilarity(train, 10)(1,886)), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(0.0)//predictKNN(train, 10)(1,1))//Prediction of item 1 for user 1 (k=10)
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
        uNeighbors = (allUsers-u).toSeq.map(x=>(x, similarityFunction(u,x))).sortWith(_._2>_._2).take(k)
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

  def getSimilarity (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {
    val allNeighbors = getNeighbors(ratings, k)
    (u,v) => allNeighbors(u).toMap.getOrElse(v, 0.0)
    // (u,v) => neighbors.getOrElse(u,Nil).toMap.getOrElse((u,v), 0.0)
  }

  def weightedSumDevKNN (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {

    // find k-nearest neighbors per user
    val allNeighbors = getNeighbors(ratings, k)
    
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    //var ratedI = computeRatedI(ratings)

    (u,i) => {
      val neighbors = allNeighbors(u)
      val ssSum = neighbors.map(_._2.abs).sum
      if (ssSum !=0) {
        neighbors.map(x=> x._2*normalizedDeviations.getOrElse((x._1,i), 0.0)).sum/ssSum
      } else 0.0
      
    }
  }

  def predictKNN(ratings : Seq[Rating], k : Int) : (Int, Int)=> Double = {
    val globalAvgValue = globalAvg(ratings)
    val usersAvgValue = usersAvg(ratings)

    val wsd = weightedSumDevKNN(ratings, k)
    (u,i) => {
      var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
      var userWSD = wsd(u, i)
      userAvg+userWSD*scale((userAvg+userWSD), userAvg)
    }
  }

  println("")
  spark.close()
}
