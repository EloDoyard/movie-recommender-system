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
    MeanAbsoluteErrorSpark(baselinePredictorSpark(train), test)
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
          "1.GlobalAvg" -> ujson.Num(getGlobalAvg(train)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(usersAvgSpark(train)(1,0)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(itemsAvgSpark(train)(0,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(itemsAvgDevSpark(train)(0,1)), // Datatype of answer: Double,
          "5.PredUser1Item1" -> ujson.Num(baselinePredictorSpark(train)(1,1)), // Datatype of answer: Double
          "6.Mae" -> ujson.Num(MeanAbsoluteErrorSpark(baselinePredictorSpark(train),test)) // Datatype of answer: Double
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

  /**
  * Compute the mean for a RDD of ratings
  * @param ratings a sequence of ratings
  * @return the mean of the RDD of ratings passed as parameter
  */
  def meanSpark(ratings : RDD[Double]) : Double = {
    ratings.sum/ratings.count()
  }

  /**
  * Compute the Mean Absolute Error of a RDD of a predictor passed as parameter according to the real values
  * @param predictor a rating prediction function taking into parameter a pair of (user, item) and returning a predicted rating
  * @param real a RDD of real ratings to evaluate the predictor on
  * @return MAE of predictor passed as parameter
  */
  def MeanAbsoluteErrorSpark(predictor : (Int, Int) => Double, real : RDD[Rating]) : Double={
    meanSpark(real.map(x=> (predictor(x.user, x.item)-x.rating).abs))
  }

  /**
  * Compute the global average rating of RDD passed as parameter
  * @param ratings a RDD of ratings
  * @return average rating
  */
  def getGlobalAvg(ratings: RDD[Rating]) : Double = meanSpark(ratings.map(_.rating))

  /**
  * Compute the average rating of each user for a RDD of ratings
  * @param ratings a RDD of ratings
  * @return map every user to its average rating
  */
  def getUsersAvg(ratings : RDD[Rating]) : Map[Int,Double] = ratings.map{
    case x : Rating => (x.user, (1, x.rating))
    }.reduceByKey((acc, a) => (acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collect().toMap

  /**
  * User Average Predictor predicting for each pair (user, item) the user's average rating
  * @param ratings a RDD of ratings
  * @return function that maps (user, item) to user's average rating
  */
  def usersAvgSpark(ratings: RDD[Rating]) : (Int, Int) => Double = {
    // global average rating
    val globalAvg = getGlobalAvg(ratings)
    // map of average rating per user
    val usersAvg = getUsersAvg(ratings)
    // returns user's average rating or global average if user not in map
    (u,i) => usersAvg.getOrElse(u, globalAvg)
  }

  /**
  * Compute the average rating of each item for a RDD of ratings
  * @param ratings a RDD of ratings
  * @return map each item to its average rating
  */
  def getItemsAvg(ratings : RDD[Rating]) : Map[Int,Double] = ratings.map{
    case x : Rating => (x.item, (1, x.rating))
  }.reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collect().toMap
  
  /**
  * Item Average Predictor predicting for each pair (user, item) the item's average rating
  * @param ratings a RDD of ratings
  * @return function that maps (user, item) to item's average rating
  */
  def itemsAvgSpark(ratings: RDD[Rating]) : (Int, Int) => Double = {
    // global average rating
    val globalAvg = getGlobalAvg(ratings)
    // map of average rating per item
    val itemsAvg = getItemsAvg(ratings)
    // returns item's average rating or global average if item not in map
    (u,i) => itemsAvg.getOrElse(i, globalAvg)
  }

  /**
  * Compute normalized deviaiton for each rating according to formula defined in handout
  * @param ratings a RDD of ratings
  * @return map (user, item) to deviation of its rating
  */
  def getNormalizedDev(ratings : RDD[Rating]) : Map[(Int,Int), Double] = {
    // global average rating
    val globalAvg = getGlobalAvg(ratings)
    // map of average rating per user
    val usersAvg = getUsersAvg(ratings)
    // map (user, item) to deviation of its rating
    ratings.map{
      case x: Rating => {
        // user's average rating or global average rating if user not in map
        val userAvg = usersAvg.getOrElse(x.user, globalAvg)
        // compute deviation
        Rating(
        x.user, x.item, ((x.rating-userAvg)/scale(x.rating, userAvg))
    )}}.groupBy(x=>(x.user, x.item)).mapValues(x=> x.toSeq.map(_.rating).head).collect().toMap
  }

  /**
  * Compute the average deviation for each item
  * @param ratings a RDD of ratings
  * @return map item to its average deviation
  */
  def getItemsAvgDev (ratings : RDD[Rating]) : Map[Int,Double] = {
    // map of (user, item) to its normalized deviation 
    val normalizedDevs = getNormalizedDev(ratings)
    // RDD of Ratings where the ratings is the normalized deviation of the original rating
    val deviation = ratings.map(x=>Rating(x.user, x.item, normalizedDevs.getOrElse((x.user, x.item), 0.0)))
    // map items to its average deviation
    deviation.map{
      case x : Rating => (x.item, (1, x.rating))
    }.reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collect().toMap
 }

 /**
  * Item Average Deviation Predictor predicting for (user, item) the item's average deviation 
  * @param ratings a RDD of ratings
  * @return function that maps (user, item) to item's average deviation
  */
 def itemsAvgDevSpark(ratings : RDD [Rating]) : (Int, Int) => Double = {
   // map items to its average deviation
  val itemsAvgDev = getItemsAvgDev(ratings)
  // returns average deviation per item or 0.0 if item not in map
   (u,i) => itemsAvgDev.getOrElse(i, 0.0)
 }

  /**
  * Baseline Predictor predicting for (user, item) prediction defined in the handout
  * @param ratings a RDD of ratings
  * @return function that maps (user, item) to its baseline prediction
  */
  def baselinePredictorSpark (ratings : RDD[Rating]) : (Int, Int) => Double = {
    // global average rating
    val globalAvg = getGlobalAvg(ratings)
    // map each user to its average rating
    val usersAvg = getUsersAvg(ratings)
    // map each item to its average deviation
    val itemsAvgDev = getItemsAvgDev(ratings)
    (u,i) => {
      // user u average rating or global average if not in map
      val userAvg = usersAvg.getOrElse(u, globalAvg)
      // item i average deviation or 0.0 if not in map
      val itemAvgDev = itemsAvgDev.getOrElse(i, 0.0)
      // baseline prediction
      (userAvg+itemAvgDev*scale((userAvg+itemAvgDev), userAvg))
    }
  }

  println("")
  spark.close()
}