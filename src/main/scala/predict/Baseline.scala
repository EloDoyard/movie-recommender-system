package predict
//package co.kbhr.scaladoc_tags

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Baseline extends App {
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
  // For these questions, data is collected in a scala Array 
  // to not depend on Spark
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()

  // global average
  val globalMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MeanAbsoluteError(computeAvgRating(train), test)
    
  }))
  val globalTime = globalMeasure.map(t => t._2) // Retrieve the timing measurements

  // user average
  val userMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MeanAbsoluteError(computeUserAvg(train), test)
    
  }))
  val userTime = userMeasure.map(t => t._2) // Retrieve the timing measurements

  // item average
  val itemMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MeanAbsoluteError(computeItemAvg(train), test)
    
  }))
  val itemTime = itemMeasure.map(t => t._2) // Retrieve the timing measurements

  // baseline
  val baselineMeasure = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    MeanAbsoluteError(computePrediction(train), test)
    
  }))
  val baselineTime = baselineMeasure.map(t => t._2) // Retrieve the timing measurements

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


      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(computeAvgRating(train)(1,1)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(computeUserAvg(train)(1,1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(computeItemAvg(train)(1,1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(computeItemAvgDev(train)(1,1)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(computePrediction(train)(1,1)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(MeanAbsoluteError(computeAvgRating(train), test)), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(MeanAbsoluteError(computeUserAvg(train), test)),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(MeanAbsoluteError(computeItemAvg(train), test)),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(MeanAbsoluteError(computePrediction(train), test))   // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(globalTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(globalTime)) // Datatype of answer: Double

          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(userTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(userTime)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(itemTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(itemTime)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(baselineTime)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(baselineTime)) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  /**
  * Compute the scale function as defined in handout
  * @param x a user's rating
  * @param y user's average rating
  * @return user's rating rescaled
  */
  def scale (x:Double,y:Double) : Double = {
    if (x>y) 5-y
    else if (x<y) y-1
    else 1
  }

  /**
  * Compute the Mean Absolute Error of a predictor passed as parameter
  * @param predictor a rating prediction function taking into parameter a user and an item 
  *   and returns prediction of rating for such pai
  * @param real Sequence of ratings to evaluate predictor on
  * @return Mean Absolute Error of predictor
  */
  def MeanAbsoluteError(predictor : (Int, Int) => Double, real : Seq[Rating]) : Double={
    mean(real.map(x=> (predictor(x.user, x.item)-x.rating).abs))
  }

  /**
  * Compute the MAE on the given data set 
  * @param data the data set on which to compute the MAE
  * @param predict function used to make a prediction for the rating
  * @return the MAE
  */
  def MAE(predict: (Int, Int)=> Double, data: Seq[Rating]): Double = {
    applyAndMean(data){
      x => (x.rating-predict(x.user, x.item)).abs
    }
  }

  /** Apply a function to every element of the data set and then average
  *  @param data the data set
  *  @param f the function to applied on each element
  *  @return The mean value computed over the data set
  */
  def applyAndMean(data: Seq[Rating])(f: (Rating=>Double)):Double={
    val res = data.foldLeft((0.0,0)){
      // The accumulator is a tuple (Double, Int) which consists of the running sum and the running count of added entities
      (acc,x) => (f(x) + acc._1, acc._2+1)
    }
    res._1/res._2
  }


  /**
  * Compute the global average rating of sequence passed as parameter
  * @param ratings a sequence of ratings
  * @return average rating
  */
  def globalAvg(ratings : Seq[Rating]) : Double = mean(ratings.map(_.rating))

  /**
  * Global Average Predictor predicting the global average predictor everytime
  * @param ratings a sequence of ratings
  * @return function of the pair (user, item) and returning a prediction for such pair
  */
  def computeAvgRating(ratings : Seq[Rating]) : (Int, Int) => Double = {
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // returning the global average rating for every value of (user, item)
    (u,i)=> globalAvgValue
  }

  /**
  * Compute average rating per user
  * @param ratings a sequence of ratings
  * @return map every user in ratings sequence to its average rating
  */
  def usersAvg (ratings : Seq [Rating]) : Map[Int, Double] = ratings.groupBy(_.user).mapValues(x=>globalAvg(x))

  /**
  * User Average Predictor predicting for each pair (user, item) the user's average rating
  * @param ratings a sequence of ratings
  * @return function that maps the pair (user, item) to the user's average rating
  */
  def computeUserAvg(ratings : Seq[Rating]) : (Int,Int) => Double = {
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // map of the average rating per user
    val usersAvgValue = usersAvg(ratings)
    // returns user's average rating or the global average if user is not in map
    (u,i) => usersAvgValue.getOrElse(u,globalAvgValue)
  }
  
  /**
  * Compute average rating per item
  * @param ratings a sequence of ratings
  * @return map every item to its average rating
  */
  def itemsAvg (ratings : Seq[Rating]) : Map[Int, Double] = ratings.groupBy(_.item).mapValues(x=>globalAvg(x))

  /**
  * Item Average Predictor predicting for (user, item) the item's average rating
  * @param ratings a sequence of ratings
  * @return function that maps (user, item) to the item's average rating
  */
  def computeItemAvg(ratings : Seq[Rating]) : (Int,Int) => Double = {
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // map of average rating per item
    val itemsAvgValue = itemsAvg(ratings)
    // returns item's average rating or global average if item not in map
    (u,i)=> itemsAvgValue.getOrElse(i, globalAvgValue)
  }

  /**
  * Compute normalized deviation for each rating according to formula defined in handout
  * @param ratings a sequence of ratings
  * @return map (user, item) to the deviation of its rating
  */
  def computeNormalizeDeviation(ratings : Seq[Rating]) : Map[(Int,Int),Double] = {
    // map of average rating per user
    val usersAvgValue = usersAvg(ratings)
    // global average rating
    val globalAvgValue = globalAvg(ratings)

    // map (user, item) to its deviation
    ratings.map(
      x=>{
        // user's average rating or global average rating if not in map
        val userAvg = usersAvgValue.getOrElse(x.user,globalAvgValue)
        // compute deviation
        Rating(
        x.user,x.item, ((x.rating-userAvg) / scale(x.rating, userAvg))
        )
      }).groupBy(x=>(x.user,x.item)).mapValues(_.head.rating)
  }

  /**
  * Compute the average deviation for each item
  * @param ratings a sequence of ratings
  * @return map item to its average deviation
  */
  def itemsAvgDev(ratings : Seq[Rating]) : Map[Int, Double] = {
    // map of normalized deviation of each (user, item)
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    // map items to its average deviation or 0 if not in map
    ratings.map(
      x=>Rating(x.user, x.item, normalizedDeviations.getOrElse((x.user, x.item), 0.0))
      ).groupBy(_.item).mapValues(x=>globalAvg(x))
  }

  /**
  * Item Average Deviation Predictor predicting for (user, item) the item's average deviation rating
  * @param ratings a sequence of ratings
  * @return function that maps (user, item) to item's average deviation rating
  */
  def computeItemAvgDev(ratings : Seq[Rating]) : (Int,Int) => Double = { 
    // map of average deviation ratings per item
    val deviationsValue = itemsAvgDev(ratings)
    // returns average deviation rating per item or 0 if not in map
    (u,i) => deviationsValue.getOrElse(i, 0.0)
  }

  /**
  * Baseline Predictor predicting for (user, item) prediction defined in the handout
  * @param ratings a sequence of ratings
  * @return function that maps (user, item) to its baseline prediction
  */
  def computePrediction(ratings : Seq[Rating]) : (Int,Int) =>Double = {
    // map of users to its average rating
    val usersAvgValue = usersAvg(ratings)
    // map of items to its item average deviation rating
    val itemsAvgDevValue = itemsAvgDev(ratings)
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // return baseline prediction
    (user: Int, item: Int) => {
      // user's average rating or global average rating if not in map
      val userAvg = usersAvgValue.getOrElse(user, globalAvgValue)
      // item's average deviation or 0 if not in map
      val itemAvgDev = itemsAvgDevValue.getOrElse(item, 0.0)  
      // baseline prediction
      (userAvg+itemAvgDev*scale((userAvg+itemAvgDev), userAvg))
    }
  }
  println("")
  spark.close()
}