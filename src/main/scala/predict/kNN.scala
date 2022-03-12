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
    Thread.sleep(1000) // Do everything here from train and test
    42        // Output answer as last value
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
          "1.k10u1v1" -> ujson.Num(getSimilarity(train, 10)(1,1)), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(0.0), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(0.0), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(0.0) // Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> List(10,30,50,100,200,300,400,800,943).map(k => 
              List(
                k,
                0.0 // Compute MAE
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

  def getNeighbors(ratings : Seq[Rating], k : Int) : Map[Int, Seq[((Int, Int), Double)]] = {
    // ratings.map(_.user).toSet.toList.permutations(2).groupBy(x._1).mapValues(
    //   y=>y.map(
    //     x=> (x, adjustedCosineSimilarityFunction(x._1, x._2))).toSeq.sortWith(_._2>_._2).take(k))

    val similarityFunction = adjustedCosineSimilarityFunction(ratings)
    val neighbors = ratings.map(_.user).toSet.toList.combinations(2).map{ case List(a, b) => (a, b) }.toSeq.foldLeft(
      Map[Int, Seq[((Int, Int), Double)]]()){
      (acc, a) => {
        acc+(
          a._1 -> (((a, similarityFunction(a._1, a._2))) +: acc.getOrElse(a._1, Seq())), 
          a._2 -> (((a, similarityFunction(a._1,a._2)))+:acc.getOrElse(a._2, Seq())))
      }
    }

    // neighbors.mapValues(x=> x._.sortWith(_._2>_._2).take(k))
    neighbors.foldLeft(Map[Int, Seq[((Int, Int), Double)]]()) {
      (acc, a) => {
        acc+(a._1 -> a._2.sortWith(_._2>_._2).take(k))
      }
    }
    // neighbors.transform{(key, value) => value.sortWith(_._2>_._2).take(k)}
  } 

  def getSimilarity (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {
    val neighbors = getNeighbors(ratings, k)
    (u,v) => {
      val uNeighborhood = neighbors.getOrElse(u,Nil).toMap
      val uvSim = uNeighborhood.getOrElse((u,v),0.0)
      if (uvSim!=0.0) uvSim 
      else {
        val vNeighborhood = neighbors.getOrElse(v, Nil).toMap
        vNeighborhood.getOrElse((v,u),0.0)
      }
  }
  }

  def weightedSumDevKNN (ratings : Seq[Rating], k : Int) : (Int, Int) => Double = {

    // find k-nearest neighbors per user
    val usersNeighbors = getNeighbors(ratings, k)
    
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    //var ratedI = computeRatedI(ratings)

    (u,i) => {
      val neighbors = usersNeighbors.getOrElse(u, Nil).toMap
      val ssSum = neighbors.mapValues(_.abs).values.sum
      if (ssSum !=0) {
        neighbors.map(x=> x._2*normalizedDeviations.getOrElse((x._1._2,i), 0.0)).sum/ssSum
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
