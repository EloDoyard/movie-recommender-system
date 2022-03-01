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
    Thread.sleep(1000) // Do everything here from train and test
    42        // Output answer as last value
  }))
  val timings_glob_avg = measurements_glob_avg.map(t => t._2) // Retrieve the timing measurements

  val measurements_user_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    Thread.sleep(1000) // Do everything here from train and test
    42        // Output answer as last value
  }))
  val timings_user_avg = measurements_user_avg.map(t => t._2) // Retrieve the timing measurements

  val measurements_item_avg = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    Thread.sleep(1000) // Do everything here from train and test
    42        // Output answer as last value
  }))
  val timings_item_avg = measurements_item_avg.map(t => t._2) // Retrieve the timing measurements

  val measurements_pred = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    Thread.sleep(1000) // Do everything here from train and test
    42        // Output answer as last value
  }))
  val timings_pred = measurements_pred.map(t => t._2) // Retrieve the timing measurements

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
          "1.GlobalAvg" -> ujson.Num(globalAvgVal), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(avgUsers(1)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(avgItems(1)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(devs(1)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(predict(devs, 1, 1, avgUsers,globalAvgVal)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(computeGlobalError(train, globalAvgVal)), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(computeUsersAvgError(train, avgUsers, globalAvgVal)),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(computeItemsAvgError(train, avgItems, globalAvgVal)),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(computeBaselineAvgError(train, avgUsers, devs, globalAvgVal))  // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_glob_avg)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_glob_avg)) // Datatype of answer: Double
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_user_avg)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_user_avg)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_item_avg)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_item_avg)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_pred)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_pred)) // Datatype of answer: Double
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

  def absoluteError(trueVal:Double, pred:Double):Double = (trueVal-pred).abs

  def computeMAE(train:Seq[Rating])(f:(Rating=>Double)):Double= 
    applyAndMean(train){x=>absoluteError(x.rating,f(x))}
    /*{
    
    val res = train.foldLeft((0.0,0))((y,x)=>(absoluteError(x.rating, f(x))+y._1, y._2+1))
    res._1/res._2
  }*/

  def applyAndMean(data:Seq[Rating])(f:(Rating=>Double)):Double={
    val res = data.foldLeft((0.0,0))((y,x)=>(f(x)+y._1, y._2+1))
    res._1/res._2
  }

  def computeGlobalError(train:Seq[Rating], globAvg:Double):Double = 
    computeMAE(train){x=>globAvg}
    /*{
    val res = train.foldLeft((0.0,0))((y,x)=>(absoluteError(x.rating,globAvg)+y._1, y._2+1))
    res._1/res._2
  }*/

  def computeItemsAvgError(train:Seq[Rating], itemsAvg:Map[Int,Double], globAvg:Double):Double = 
    computeMAE(train){y=>itemAvg(itemsAvg, y.item, globAvg)}
    /*{
    val res = train.foldLeft((0.0,0))((y,x)=>(absoluteError(x.rating,itemAvg(itemsAvg, x.item, globAvg))+y._1, y._2+1))
    res._1/res._2
  }*/
    
  def computeUsersAvgError(train:Seq[Rating], usersAvg:Map[Int,Double], globAvg:Double):Double = 
    computeMAE(train){y=>userAvg(usersAvg, y.user, globAvg)}
    /*{
    val res = train.foldLeft((0.0,0))((y,x)=>(absoluteError(x.rating, userAvg(usersAvg, x.user, globAvg))+y._1, y._2+1))
    res._1/res._2
  }*/

  def computeBaselineAvgError(train:Seq[Rating],usersAvg:Map[Int,Double], devs:Map[Int,Double],globAvg:Double):Double =
    computeMAE(train){y=> predict(devs, y.user, y.item, usersAvg, globAvg)}
    /*{
    val res = train.foldLeft((0.0,0))((y,x)=>(absoluteError(x.rating, predict(devs, x.user, x.item, usersAvg,globAvg))+y._1, y._2+1))
    res._1/res._2
  }*/

  def computeAllUsersAvg(train:Seq[Rating]):Map[Int, Double] = 
    train.groupBy(_.user).mapValues(x=> applyAndMean(x){y=>y.rating})
  
  def computeAllItemsAvg(train:Seq[Rating]):Map[Int, Double] = 
    train.groupBy(_.item).mapValues(x=> applyAndMean(x){y=>y.rating})

  def globalAvg(train:Seq[Rating]):Double = 
    applyAndMean(train){x=>x.rating}
    //mean(train.map(_.rating))

  def userAvg(usersAvg:Map[Int,Double], userId:Int, globAvg:Double):Double = 
    usersAvg.getOrElse(userId, globAvg)

  def itemAvg(itemsAvg:Map[Int,Double], itemId:Int, globAvg:Double):Double = 
    itemsAvg.getOrElse(itemId, globAvg)

  def computeAllDevs(train:Seq[Rating], usAvg:Map[Int,Double]):Map[Int,Double] = {
    train.groupBy(y=>y.item)
    .mapValues{
      y=>applyAndMean(y){
        x=> (x.rating-usAvg(x.user))/scale(x.rating, usAvg(x.user))
      }
      /*y=>mean(y.map({
        x=> (x.rating-usAvg(x.user))/scale(x.rating, usAvg(x.user))
      }))*/
    }
  }
  
  def predict(devs:Map[Int,Double], userId:Int, itemId:Int, usAvg:Map[Int,Double], globAvg:Double):Double = {
    if (devs.contains(itemId) && usAvg.contains(userId)) {
      usAvg(userId) + devs(itemId) *scale(devs(itemId) + usAvg(userId), usAvg(userId))
    }else{
      if(usAvg.contains(userId)){
        usAvg(userId)
      }else{
        globAvg 
      }
    }
    
  }

  def scale(rat:Double, usAvg:Double):Double ={
    if (rat > usAvg){
      5-usAvg
    }else if (rat < usAvg){
      usAvg-1
    }else{
      1.0
    }
  }
  
}
