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
      // var predUser1Item1 = PredUser1Item1()
      // val predictor1Sim = predict(train, (_,_)=>1)
      val ACSim = adjustedCosineSimilarityFunction(train)
      // val predictorACSim = predict(train, ACSim)
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "P.1" -> ujson.Obj(
          "1.PredUser1Item1" -> ujson.Num(0.0),//predictor1Sim(1,0)), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(0.0)//MeanAbsoluteError(0.0)//predictor1Sim, test))         // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          "1.AdjustedCosineUser1User2" -> ujson.Num(ACSim(2, 1)), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(0.0),//predictorACSim(1,1)),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(0.0)//MeanAbsoluteError(predictorACSim, test)) // MAE when using adjusted cosine similarity
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

  // similarity function that takes as parameter 2 users u and v
  // weighted sum deviation that takes as parameter function of similarity and return a function of user
  // prediction function

  def adjustedCosineSimilarityFunction (ratings : Seq[Rating]) : (Int, Int)=> Double = {
    // function of preprocessed ratings
    val preprocessedRatings = preprocessedRating(ratings)
    
    (u,v) => {  
      // tuple of (list of ratings by u, list of ratings by v, set of items rated by u, set of items rated by v)
      var ratedByUV = ratings.foldLeft((List[Rating](), List[Rating](), Set[Int](),Set[Int]())){ 
        (a,b) => {
          if (b.user == u) {
              (a._1:+ b, a._2,a._3+b.item, a._4)
          } else if (b.user == v) {
            (a._1, a._2:+ b,a._3, a._4+b.item)
          } else a
        }
      }
      // set of items rated by both u and v
      val ratedByBoth = ratedByUV._3.intersect(ratedByUV._4)
      // println(ratedByBoth)

      // println(ratedByBoth.length)
      // filter on both list of ratings the ratings where the item was rated by both users
      // we sort by item the ratings for each list
      // val uRatingsOfItemRatedByBoth = ratedByUV._1.filter(x=> ratedByBoth.contains(x.item)).sortBy(_.item)
      // val vRatingsOfItemRatedByBoth = ratedByUV._2.filter(x=> ratedByBoth.contains(x.item)).sortBy(_.item)

      // compute similarity between users u and v
      ratedByBoth.map(
        i => (preprocessedRatings.getOrElse((u,i), 0.0), preprocessedRatings.getOrElse((v,i),0.0))).map{
          case (x,y)=> x*y}.sum
    }
  }

  def weights (ratings : Seq[Rating]) : Map[Int, Double] = {
    val normalizedDev = computeNormalizeDeviation(ratings)

    ratings.map(
      x=>Rating(x.user, x.item, math.pow(normalizedDev.getOrElse((x.user, x.item), 0.0),2))
      ).groupBy( _.user).mapValues(
          y=>math.sqrt(y.map(_.rating).sum))
  }

  def preprocessedRating(ratings:Seq[Rating]): Map[(Int,Int),Double] = {
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    val usersWeights = weights(ratings)

    ratings.map(x=> {
      val weight = usersWeights.getOrElse(x.user, 0.0)
      if (weight!= 0.0) Rating (x.user, x.item, normalizedDeviations.getOrElse((x.user, x.item), 0.0)/weight)
      else Rating(x.user, x.item, 0.0)
    }).groupBy(x=>(x.user, x.item)).mapValues(_.head.rating)
    // (u,i) => {
    //   var ratedByU = ratings.filter(x=>x.user == u)
    //   val denominator = math.sqrt(ratedByU.map(x=>math.pow(itemsDev(x.user,x.item),2)).sum)
    //   val rating = itemsDev(u,i)
    //   if (rating !=0 && !ratedByU.isEmpty) rating/denominator
    //   else 0
    // }
  }

  // def preProcRatNum(rats:Seq[Rating]): (Int, Int)=>Double = {
  //   (u1, u2) => {
  //     var ratedByUs = rats.foldLeft((List[Rating](), List[Rating](), Set[Int](),Set[Int]())){
  //       (a,b) => {
  //         if (b.user == u1) {
  //             (a._1:+ b, a._2,a._3+b.item, a._4)
  //         } else if (b.user == u2) {
  //           (a._1, a._2:+ b,a._3, a._4+b.item)
  //         } else a
  //       }
  //     }
  //     // set
  //     var ratedByBoth = ratedByUs._3.intersect(ratedByUs._4)
  //     var usRatesCommun = ratedByUs._1.filter(x=> ratedByBoth.contains(x.item)).sortBy(_.item)
  //     var vsRatesCommun = ratedByUs._2.filter(x=> ratedByBoth.contains(x.item)).sortBy(_.item)

  //     val denominator = preProcRatDen(usRatesCommun)
  //     ratedByBoth.map(x => (denominator(x), denominator(x))).map{case (x,y)=> x*y}.sum
  //   }
  // }

  // def itemsDeviation(ratings : Seq[Rating]) : (Int,Int) => Double = {
  //   // val normalizedDeviations = computeNormalizeDeviation(ratings)
  //   (u,i) => {
  //     val userRatings = ratings.filter(x=>x.user==u)
  //     val userAvg = mean(userRatings.map(_.rating))
  //     val uRatedI = userRatings.filter(x=>x.item==i)
  //     if (!uRatedI.isEmpty) {
  //       val rating = uRatedI.map(_.rating).head
  //       (rating-userAvg)/scale(rating, userAvg)
  //     } else 0 
  //   }
  // }

  def weightedSumDeviation (ratings : Seq[Rating], i : Int, similarityFunction : (Int,Int)=>Double) : Int => Double = {
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    var ratedI = ratings.withFilter(x=> x.item == i)
    a => {
      var ss = ratedI.map(x=>(x.user, similarityFunction(x.user, a))).toMap
      var ssSum = ss.values.map(_.abs).sum
      if (ssSum!=0.0){//au moins un element de ss n'est pas null 
        ratedI.map(
          x=> ss.getOrElse(x.user,0.0)*normalizedDeviations.getOrElse((x.user, i), 0.0)).sum / ssSum
      } else 0.0
    }
  }

  def predict(ratings : Seq[Rating], similarityFunction:(Int,Int)=>Double) : (Int, Int)=>Double = {
    val globalAvgValue = globalAvg(ratings)
    val usersAvgValue = usersAvg(ratings)
    (u,i) => {
      var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
      var userWSD = weightedSumDeviation(ratings, i,similarityFunction)(u)
      userAvg+userWSD*scale((userAvg+userWSD), userAvg)
    } 
  }

  println("")
  spark.close()
}
