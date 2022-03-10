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

  //B1
  // val avgRating = computeAvgRating(train)
  // val usersAvg = computeUserAvg(train)
  // val itemsAvg = computeItemAvg(train)
  // val normalizedDeviations = computeNormalizeDeviation(train)
  // val itemsAvgDev = computeItemAvgDev(train)
  // val predictions = computePrediction(train)
  // // val predictor = predict(train)
  // val globalPrediction = globalAvgPredictor()
  // val userAvgPrediction = userAvgPredictor()
  // val itemAvgPrediction = itemAvgPredictor()
  // val baselinePrediction = baselinePredictor()
  //B3
  //val deviationPred = test.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg))))
  //var devAvgPred = deviationPred.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))

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

  def scale (x:Double,y:Double) : Double = {
    if (x>y) 5-y
    else if (x<y) y-1
    else 1
  }

  def MeanAbsoluteError(predictor : (Int, Int) => Double, real : Seq[Rating]) : Double={
    mean(real.map(x=> (predictor(x.user, x.item)-x.rating).abs))
  }

  def globalAvg(ratings : Seq[Rating]) : Double = mean(ratings.map(_.rating))

  def computeAvgRating(ratings : Seq[Rating]) : (Int, Int) => Double = {
    val globalAvgValue = globalAvg(ratings)
    (u,i)=> globalAvgValue
  }

  def usersAvg (ratings : Seq [Rating]) : Map[Int, Double] = ratings.groupBy(_.user).mapValues(x=>globalAvg(x))

  def computeUserAvg(ratings : Seq[Rating]) : (Int,Int) => Double = {// Map[Int,Double] = {
    // ratings.groupBy(_.user).mapValues(x=>mean(x.map(_.rating)))
    val globalAvgValue = globalAvg(ratings)
    val usersAvgValue = usersAvg(ratings)
    (u,i) => usersAvgValue.getOrElse(u,globalAvgValue)
  }

  def itemsAvg (ratings : Seq[Rating]) : Map[Int, Double] = ratings.groupBy(_.item).mapValues(x=>globalAvg(x))

  def computeItemAvg(ratings : Seq[Rating]) : (Int,Int) => Double = { //Map[Int,Double] = {
    // ratings.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
    val globalAvgValue = globalAvg(ratings)
    val itemsAvgValue = itemsAvg(ratings)
    (u,i)=> itemsAvgValue.getOrElse(i, globalAvgValue)
  }

  def computeNormalizeDeviation(ratings : Seq[Rating]) : Map[(Int,Int),Double] = { //Map[(Int,Int), Double] = {
    val usersAvgValue = usersAvg(ratings)
    val globalAvgValue = globalAvg(ratings)

    // (u,i) => {
    //   val userAvg = usersAvgValue.getOrElse(u, globalAvgValue)
    //   val dev = ratings.filter(x=> x.user == u && x.item == i).map(
    //     x=> (x.rating-userAvg)/scale(x.rating, userAvg))
    //   if (!dev.isEmpty) dev.head
    //   else 0.0
      ratings.map(
      x=>{
        val userAvg = usersAvgValue.getOrElse(x.user,globalAvgValue)
        Rating(
        x.user,x.item, ((x.rating-userAvg) / scale(x.rating, userAvg))
        )
      }).groupBy(x=>(x.user,x.item)).mapValues(_.head.rating)
    //}
  }

  def itemsAvgDev(ratings : Seq[Rating]) : Map[Int, Double] = {
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    val globalAvgValue = globalAvg(ratings)
    val deviationsValue = ratings.map(x=>Rating(x.user, x.item, normalizedDeviations.getOrElse((x.user, x.item), 0.0)))
    deviationsValue.groupBy(_.item).mapValues(x=>globalAvg(x))
  }
  def computeItemAvgDev(ratings : Seq[Rating]) : (Int,Int) => Double = { // Map[Int,Double] = {
    // determine U(i) = set of users with rating for i
    // do mean of noramlized deviation of ratings
    // val normalizedDeviations = computeNormalizeDeviation(ratings)
    // val globalAvgValue = globalAvg(ratings)
    // val deviationsValue = ratings.map(x=>Rating(x.user, x.item, normalizedDeviations.getOrElse((x.user, x.item), 0.0)))
    
    val deviationsValue = itemsAvgDev(ratings)
    (u,i) => deviationsValue.getOrElse(i, 0.0)//{
    //   val itemDev=ratings.filter(x=>x.item == i).map(
    //     x=>normalizedDeviations(x.user, x.item)).head

    //   if (!itemDev.isEmpty) globalAvgValue
    //   else 0.0
    // }
    // ratings.map(
    //   x=>Rating(x.user, x.item, normalizedDeviations(x.user, x.item)
    //   )).groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
    // mean(deviationsValue.filter(x=>x.item == i).map(x=>x.rating))
  }

  def computePrediction(ratings : Seq[Rating]) : (Int,Int) =>Double = {
    val usersAvgValue = usersAvg(ratings)
    val itemsAvgDevValue = itemsAvgDev(ratings)
    val globalAvgValue = globalAvg(ratings)
    (user: Int, item: Int) => {
      val userAvg = usersAvgValue.getOrElse(user, globalAvgValue) // usersAvg.getOrElse(user,avgRating)
      val itemAvgDev = itemsAvgDevValue.getOrElse(item, 0.0)  //itemsAvgDev.getOrElse(item, 0.0)
      (userAvg+itemAvgDev*scale((userAvg+itemAvgDev), userAvg))
    }
  }

 /////////////////////////////////////////////////////////////////////////////

  // def globalAvgPredictor() : (Int, Int) => Double = {
  //   println("computing globalAvgPredictor ")
  //   (u,i) => avgRating
  // }
  // def globalAvgPredictor(ratings : Seq[Rating]) : (Int, Int) => Double = {
  //   (a,b) => computeAvgRating(ratings)
  // }

  // def userAvgPredictor() : (Int, Int) => Double = {
  //   println("computing userAvgPredictor ")
  //   (u,i) => usersAvg.getOrElse(u,avgRating)
  // }
  // def userAvgPredictor(ratings : Seq[Rating]) : (Int, Int) => Double = {
  //   var globalAvg = mean(ratings.map(x => x.rating))
  //   var usersAvg = ratings.groupBy(_.user).mapValues(x=>mean(x.map(_.rating)))
  //   (u,i) => usersAvg.getOrElse(u, globalAvg)
  // }

  // def itemAvgPredictor () : (Int, Int) => Double =  {
  //   println("computing itemAvgPredictor ")
  //   (u,i) => itemsAvg.getOrElse(i, avgRating)
  // }
  // def itemAvgPredictor (ratings : Seq[Rating]) : (Int, Int) => Double = {
  //   var globalAvg = mean(ratings.map(x => x.rating))
  //   var itemsAvg = ratings.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
  //   (u,i) => itemsAvg.getOrElse(i, globalAvg)
  // }

  // def baselinePredictor (ratings : Seq[Rating], to_pred:Seq[Rating]) : (Int, Int) => Double = {
  //   var globalAvg = mean(ratings.map(x => x.rating))
  //   var usersAvg = ratings.groupBy(_.user).mapValues(x=>mean(x.map(_.rating)))
  //   var itemsAvg = ratings.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
  //   // potentiellement faut de faire map sur to_pred
  //   // val deviationPred = ratings.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user))/scale(x.rating, usersAvg.getOrElse(x.user)))) 
  //   val deviationPred = to_pred.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg)))) 
  //   var devAvgPred = deviationPred.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
  //   (u,i) => usersAvg.getOrElse(u, globalAvg)+devAvgPred(i)*scale(usersAvg.getOrElse(u, globalAvg)+devAvgPred(i), usersAvg.getOrElse(u, globalAvg))
  // }

////////////////////////////////////////////////////////////////////////////

  // def baselinePredictor() : (Int, Int)=>Double = {
  //   println("computing baselinePredictor ")
  //   (u,i) => predictions(u,i)
  // }
  // def baselinePredictor (ratings : Seq[Rating]) : (Int, Int) => Double = {
  //   val copy = ratings
  //   var globalAvg = mean(ratings.map(x => x.rating))
  //   var usersAvg = ratings.groupBy(_.user).mapValues(x=>mean(x.map(_.rating)))
  //   var itemsAvg = ratings.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
  //   // potentiellement faut de faire map sur to_pred
  //   val deviationPred = copy.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg)))) 
  //   // val deviationPred = to_pred.map(x=>Rating(x.user, x.item, (x.rating-usersAvg.getOrElse(x.user, globalAvg))/scale(x.rating, usersAvg.getOrElse(x.user, globalAvg)))) 
  //   var devAvgPred = deviationPred.groupBy(_.item).mapValues(x=>mean(x.map(_.rating)))
  //   (u,i) => usersAvg.getOrElse(u, globalAvg)+devAvgPred(i)*scale(usersAvg.getOrElse(u, globalAvg)+devAvgPred(i), usersAvg.getOrElse(u, globalAvg))
  // }
  println("")
  spark.close()
}