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
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "P.1" -> ujson.Obj(
          "1.PredUser1Item1" -> ujson.Num(predict(train, similarityOne)(1,1)), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(MeanAbsoluteError(predict(train, similarityOne), test))         // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          "1.AdjustedCosineUser1User2" -> ujson.Num(adjustedCosineSimilarityFunction(train)(2, 1)), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(predict(train, adjustedCosineSimilarityFunction(train))(1,1)),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(MAE(predict(train, adjustedCosineSimilarityFunction(train)), test)) // MAE when using adjusted cosine similarity
        ),
        "P.3" -> ujson.Obj(
          "1.JaccardUser1User2" -> ujson.Num(jaccardCoefficient(train)(1,2)), // Similarity between user 1 and user 2 (jaccard similarity)
          "2.PredUser1Item1" -> ujson.Num(predict(train, jaccardCoefficient(train))(1,1)),  // Prediction item 1 for user 1 (jaccard)
          "3.JaccardPersonalizedMAE" -> ujson.Num(MAE(predict(train, jaccardCoefficient(train)), test)) // MAE when using jaccard similarity
        )
      )
      val json = write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }


  /**
  * Similarity of 1 whatever pair of users is passed as arguemnt
  * @param u1 user 1
  $ @param u2 user 2
  * @return similarity of 1
  */
  def similarityOne = (u1 : Int, u2 : Int) => 1.0

  /**
  * Compute the adjusted Cosine Similarity function for sequence of ratings using method described in handout
  * @param ratings a sequence of ratings
  * @return function that map for each pair of users its adjusted cosine similarity
  */
  def adjustedCosineSimilarityFunction (ratings : Seq[Rating]) : (Int, Int)=> Double = {
    // map each pair (user, item) to its pre-processed rating
    val preprocessedRatings = preprocessedRating(ratings)
    
    (u,v) => {  
      // tuple of (list of ratings by u, list of ratings by v, set of items rated by u, set of items rated by v)
      var ratedByUV = ratings.foldLeft((List[Rating](), List[Rating](), Set[Int](),Set[Int]())){ 
        (a,b) => {
          if (b.user == u && b.user == v) {
            //if rating was made by both u and v, add rating and item's rating to accumulator
            (a._1:+b, a._2:+b, a._3+b.item, a._4+b.item)
          } else if (b.user == u) {
            // add u's rating and rating's item to respectively u's rating list and u's set of rated items
            (a._1:+ b, a._2,a._3+b.item, a._4)
          } else if (b.user == v) {
            // add v's rating and rating's item to respectively v's rating list and v's set of rated items
            (a._1, a._2:+ b,a._3, a._4+b.item)
          } else a
        }
      }
      // set of items rated by both u and v
      val ratedByBoth = ratedByUV._3.intersect(ratedByUV._4)

      // compute similarity between users u and v
      ratedByBoth.toSeq.map(
        i => (preprocessedRatings.getOrElse((u,i), 0.0), preprocessedRatings.getOrElse((v,i),0.0))).map{
          case (x,y)=> x*y}.sum
    }
  }

  /**
  * Map each user to sequence of all of his ratings
  * @param ratings a sequence of ratings
  * @return map user to its sequence of ratings
  */
  def computeRatedByU (ratings : Seq [Rating]) : Map[Int, Seq [Rating]] = ratings.groupBy(_.user)

  /**
  * Compute the Jaccard Coefficient function
  * @param ratings a sequence of ratings
  * @return map pair of users (u,v) to its jaccard coefficient
  */
  def jaccardCoefficient (ratings : Seq[Rating]) : (Int, Int) => Double = {
    // map every user to sequence of its ratings
    val ratedByUs = computeRatedByU(ratings)
    (u,v) => {
      // get sequence of ratings for each user u and v
      val uRatings = ratedByUs.getOrElse(u, Nil)
      val vRatings = ratedByUs.getOrElse(v, Nil)

      // items rated by both users
      val ratedByBoth = uRatings.map(_.item).toSet.intersect(vRatings.map(_.item).toSet)
      // number of items rated by both users
      val sizeIntersection = ratedByBoth.size 
      // compute jaccard coefficient
      sizeIntersection/(uRatings.size+vRatings.size-sizeIntersection)
    }
  }

  /**
  * Compute the weight for each user in the similarity metric
  * @param ratings a sequence of ratings
  * @return map each user to its weight
  */
  def weights (ratings : Seq[Rating]) : Map[Int, Double] = {
    // Map each pair (user, item) to its normalized deviation rating
    val normalizedDev = computeNormalizeDeviation(ratings)

    // compute weight, the squared root of the sum of squared normalized deviation ratings of a user
    ratings.map(
      x=>Rating(x.user, x.item, math.pow(normalizedDev.getOrElse((x.user, x.item), 0.0),2))
      ).groupBy( _.user).mapValues(
          y=>math.sqrt(y.map(_.rating).sum))
  }

  /**
  * Map each (user, item) to its the preprocessed rating as described in equation 9 of the handout
  * @param ratings a sequence of ratings
  * @return map (user, item) to its preprocessed rating
  */
  def preprocessedRating(ratings:Seq[Rating]): Map[(Int,Int),Double] = {
    // map (user, item) to its normalized deviation rating
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    // map user to its weights in preprocessed rating
    val usersWeights = weights(ratings)

    // compute preprocessed rating as defined in handout
    ratings.map(x=> {
      val weight = usersWeights.getOrElse(x.user, 0.0)
      if (weight!= 0.0) Rating (x.user, x.item, normalizedDeviations.getOrElse((x.user, x.item), 0.0)/weight)
      else Rating(x.user, x.item, 0.0)
    }).groupBy(x=>(x.user, x.item)).mapValues(_.head.rating)
  }

  /**
  * Map each item to a sequence of all of its ratings
  * @param ratings a sequence of ratings
  * @return map item to sequence of ratings
  */
  def computeRatedI (ratings : Seq[Rating]) : Map[Int, Seq[Rating]] = ratings.groupBy(_.item)

  /**
  * Compute user-specific weighted-sum deviation function as defined in the handout, equation 7
  * @param ratings a sequence of ratings
  * @param similarityFunction the similarity function to use when computing the weighted sum
  * @return map each (user, item) to its weighted-sum deviation
  */
  def weightedSumDeviation (ratings : Seq[Rating], similarityFunction : (Int,Int)=>Double) : (Int, Int) => Double = {
    // Map (user, item) to its normalized deviation rating
    val normalizedDeviations = computeNormalizeDeviation(ratings)
    // Map item to sequence of all of its ratings
    var ratedI = computeRatedI(ratings)

    (u, i) => {
      // get i's ratings
      val ratedIUsers = ratedI.getOrElse(i, Nil)
      // get similarity values between user u and all others that rated i
      var ss = ratedIUsers.map(x=>x.user-> similarityFunction(x.user, u)).toMap
      // compute denominator of weighted sum
      var ssSum = ss.mapValues(_.abs).values.sum
      // weighted sum deviation
      if (ssSum!=0.0){
        ratedIUsers.map(
          x=> ss.getOrElse(x.user,0.0)*normalizedDeviations.getOrElse((x.user, i), 0.0)).sum / ssSum
      } else 0.0
    }
  }

  /**
  * Personnalized Predictor predicting for (user, item) the baseline prediction 
  *   and using the user-specific weighted-sum deviation as deviation
  * @param ratings a sequence of ratings
  * @return function that maps (user, item) to its prediction
  */
  def predict(ratings : Seq[Rating], similarityFunction : (Int,Int)=>Double) : (Int, Int)=>Double = {
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // map each user to its average rating
    val usersAvgValue = usersAvg(ratings)
    // map each user to its weighted-sum deviation
    val wsd = weightedSumDeviation(ratings, similarityFunction)

    (u,i) => {
      // u average rating
      var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
      // u weighted-sum deviation
      var userWSD = wsd(u, i)
      // prediction
      userAvg+userWSD*scale((userAvg+userWSD), userAvg)
    } 
  }

  println("")
  spark.close()
}
