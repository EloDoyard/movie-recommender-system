package predict

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

  val measurements_glob_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    computeGlobalMAE(train, test)
  }))
  val timings_glob_avg = measurements_glob_avg.map(t => t._2)

  val measurements_user_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    computeUsersMAE(train, test)
  }))
  val timings_user_avg = measurements_user_avg.map(t => t._2)

  val measurements_item_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    computeItemsMAE(train, test)
  }))
  val timings_item_avg = measurements_item_avg.map(t => t._2)

  val measurements_pred = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    computeBaselineMAE(train, test)
  }))
  val timings_pred = measurements_pred.map(t => t._2)

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
      val avgUsers = computeAllUsersAvg(train)
      val avgItems = computeAllItemsAvg(train)
      val globalAvgVal = globalAvg(train)
      val devs = computeAllDevs(train,avgUsers)

      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(globalAvgVal), 
          "2.User1Avg" -> ujson.Num(avgUsers(1)),  
          "3.Item1Avg" -> ujson.Num(avgItems(1)),   
          "4.Item1AvgDev" -> ujson.Num(devs(1)), 
          "5.PredUser1Item1" -> ujson.Num(predict(devs, 1, 1, avgUsers, globalAvgVal)) 
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(computeGlobalMAE(train, test)), 
          "2.UserAvgMAE" -> ujson.Num(computeUsersMAE(train, test)),  
          "3.ItemAvgMAE" -> ujson.Num(computeItemsMAE(train, test)),   
          "4.BaselineMAE" -> ujson.Num(computeBaselineMAE(train, test))  
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_glob_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_glob_avg)) 
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_user_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_user_avg)) 
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_item_avg)), 
            "stddev (ms)" -> ujson.Num(std(timings_item_avg)) 
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_pred)), 
            "stddev (ms)" -> ujson.Num(std(timings_pred)) 
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  println("")
  spark.close()

  /** Compute the MAE over the test set using the global average as prediction
  *
  *  @param train the train data set 
  *  @param test the test data set
  *  @return The MAE
  */
  def computeGlobalMAE(train: Seq[Rating],test: Seq[Rating]):Double = {
    val globalAvgValue = globalAvg(train)
    computeMAE(test){x=>globalAvgValue}
  }

  /** Compute the MAE over the test set using the item average as prediction
  *
  *  @param train the train data set 
  *  @param test the test data set
  *  @return The MAE
  */
  def computeItemsMAE(train: Seq[Rating], test: Seq[Rating]):Double = {
    // Computed only if needed
    lazy val globalAvgValue = globalAvg(train)
    val itemsAvg = computeAllItemsAvg(train)

    computeMAE(test){
      y => itemAvg(itemsAvg, y.item, globalAvgValue)
    }
  }

  /** Compute the MAE over the test set using the user average as prediction
  *
  *  @param train the train data set 
  *  @param test the test data set
  *  @return The MAE
  */
  def computeUsersMAE(train: Seq[Rating], test: Seq[Rating]):Double = {
    // Computed only if needed
    lazy val globalAvgValue = globalAvg(train)
    val usersAvg = computeAllUsersAvg(train)
    computeMAE(test){
      y => userAvg(usersAvg, y.user, globalAvgValue)
    }
  }

  /** Compute the MAE over the test set using the prediction formula
  *
  *  @param train the train data set 
  *  @param test the test data set
  *  @return The MAE
  */
  def computeBaselineMAE(train: Seq[Rating], test: Seq[Rating]):Double = {
    // Computed only if needed
    lazy val globalAvgValue = globalAvg(train)

    val usersAvg = computeAllUsersAvg(train)
    val devs = computeAllDevs(train, usersAvg)

    computeMAE(test){
      y => predict(devs, y.user, y.item, usersAvg, globalAvgValue)
    }
  }

  /** Compute the Mean Absolute Error between the actual rating and the computed one (using the given formula)
  *  @param data the data set on which to compute the MAE 
  *  @param f the prediction function used
  *  @return The MAE
  */
  def computeMAE(data: Seq[Rating])(f: (Rating=>Double)):Double = 
    applyAndMean(data){
      x => (x.rating-f(x)).abs
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

  /** Group the element in the data set with the given key function. Then for each group, apply the asked to each element and compute the mean value over the element of the group
  *  @param data the data set
  *  @param key the key on which we need to group the Ratings
  *  @param f the function to apply on each Rating
  *  @return map between all the keys in the training data set to their mean computed value
  */
  def groupByApplyMean(data: Seq[Rating], key: (Rating=>Int))(f: (Rating=>Double)): Map[Int,Double] = 
    data.foldLeft(Map[Int, (Double, Int)]()){
      // The accumulator is a map mapping the key (an int) to a pair (Double, Int) corresponding to the running sum of the value we want to compute and the number of computed value respectively 
      (acc, x)=>{
        // We access the value already stored or get 0 if no value was stored for this key
        val cur = acc.getOrElse(key(x), (0.0, 0))
        // Update of the map
        acc + (key(x) -> (cur._1+f(x), cur._2+1))
      }
    }.mapValues{
      // For each key in the hashmap, we compute the mean of the computed value
      x=>x._1/x._2
    }

  /** Compute the average rating for each user
  *  @param data the data set
  *  @return map between user to her.his average rating
  */
  def computeAllUsersAvg(data: Seq[Rating]):Map[Int, Double] =  groupByApplyMean(data, x=>x.user)(_.rating)
  
  /** Compute the average rating for each item
  *  @param data the data set
  *  @return map between item to its average rating
  */
  def computeAllItemsAvg(data: Seq[Rating]):Map[Int, Double] = groupByApplyMean(data, x=>x.item)(_.rating)

  /** Compute the total average over the whole dataset
  *  @param data the data set
  *  @return global average over the data set
  */
  def globalAvg(data: Seq[Rating]):Double = applyAndMean(data)(_.rating)

  /** Get the average rating for the requested user in the given map or the global average if the user is not in the training data set
  *  @param usersAvg the map from user id to their average rating
  *  @param userId the user of which we want to retrieve the rating average
  *  @param globAvg (by value) the global average of the training data set
  *  @return user rating average
  */
  def userAvg(usersAvg: Map[Int,Double], userId: Int, globAvg: => Double):Double = usersAvg.getOrElse(userId, globAvg)

  /** Get the average rating for the requested item in the given map or the global average if the item is not in the training data set
  *  @param itemsAvg the map from item id to their average rating
  *  @param itemId the item of which we want to retrieve the rating average
  *  @param globAvg (by value) the global average of the training data set
  *  @return item rating average
  */
  def itemAvg(itemsAvg: Map[Int,Double], itemId: Int, globAvg: => Double):Double = itemsAvg.getOrElse(itemId, globAvg)

  /** Compute the deviation with the formula given in the handout
  *  @param data the training data set
  *  @param usAvg map between user and his.her average rating in the training data set 
  *  @return map between items and their deviance
  */
  def computeAllDevs(data: Seq[Rating], usAvg: Map[Int,Double]): Map[Int,Double] = groupByApplyMean(data, x=>x.item){
    x => {
        val avg = usAvg(x.user)
        (x.rating-avg)/scale(x.rating, avg)
    }
  }

  /** Compute the prediction with the formula given in the handout
  *  @param devs map between an item and its previously computed deviation
  *  @param userId the user for which we want to compute the prediction 
  *  @param itemId the item for which we want to compute the prediction
  *  @param usAvg map between user and his.her average rating in the training data set 
  *  @return the predicted rating
  */
  def predict(devs: Map[Int,Double], userId: Int, itemId: Int, usAvg: Map[Int,Double], globAvg: => Double):Double = {
    val dev = devs.getOrElse(itemId, 0.0)
    val avg = usAvg.getOrElse(userId, globAvg)
    avg + dev*scale(dev + avg, avg)
  }

  /** Compute the scale given in the handout 
  *  @param rat the first argument of the scale function
  *  @param usAvg the average value for the user
  *  @return the scale value
  */
  def scale(rat: Double, usAvg: Double):Double = {
    if (rat > usAvg){
      5-usAvg
    }else if (rat < usAvg){
      usAvg-1
    }else{
      1.0
    }
  }
}