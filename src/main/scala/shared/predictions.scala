package shared
import org.apache.spark.rdd.RDD
import scala.math
import scala.util.Sorting


package object predictions
{
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f : ()=>Double ) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end-start)/1000000.0)
  }

  def mean(s :Seq[Double]): Double =  if (s.size > 0) s.reduce(_+_) / s.length else 0.0
  def std(s :Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else {
      val m = mean(s)
      scala.math.sqrt(s.map(x => scala.math.pow(m-x, 2)).sum / s.length.toDouble)
    }
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def load(spark : org.apache.spark.sql.SparkSession,  path : String, sep : String) : org.apache.spark.rdd.RDD[Rating] = {
    val file = spark.sparkContext.textFile(path)
    return file
      .map(l => {
        val cols = l.split(sep).map(_.trim)
        toInt(cols(0)) match {
          case Some(_) => Some(Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble))
          case None => None
        }
    })
      .filter({ case Some(_) => true 
                case None => false })
      .map({ case Some(x) => x 
            case None => Rating(-1, -1, -1)})
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
    
    // cache
    var preds = Map[(Int, Int), Double]()

    // return baseline prediction
    (user: Int, item: Int) => {
      // get value if already in cache
      var pred = preds.getOrElse((user, item), 0.0)
      if (pred == 0.0) {
        // user's average rating or global average rating if not in map
        val userAvg = usersAvgValue.getOrElse(user, globalAvgValue)
        // item's average deviation or 0 if not in map
        val itemAvgDev = itemsAvgDevValue.getOrElse(item, 0.0)  
        // baseline prediction
        pred = (userAvg+itemAvgDev*scale((userAvg+itemAvgDev), userAvg))
        // add value to cache
        preds = preds+((user,item)->pred)
      }
      // return prediction
      pred
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
  def getNormalizedDev(ratings : RDD[Rating]) : RDD[(Int, (Int,Double))] = {
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
        (x.item, (1,(x.rating-userAvg)/scale(x.rating, userAvg)))
    }}//.groupBy(x=>(x.user, x.item)).mapValues(x=> x.toSeq.map(_.rating).head).collect().toMap
  }

  /**
  * Compute the average deviation for each item
  * @param ratings a RDD of ratings
  * @return map item to its average deviation
  */
  def getItemsAvgDev (ratings : RDD[Rating]) : Map[Int,Double] = {
     // global average rating
    val globalAvg = getGlobalAvg(ratings)
    // map of average rating per user
    val usersAvg = getUsersAvg(ratings)

    // val normalizedDevs = 
    getNormalizedDev(ratings).reduceByKey((acc,a)=>(acc._1+a._1, acc._2+a._2)).mapValues(x=>x._2/x._1).collect().toMap
    // RDD of Ratings where the ratings is the normalized deviation of the original rating
    // val deviation = ratings.map(x=>Rating(x.user, x.item, normalizedDevs.getOrElse((x.user, x.item), 0.0)))
    // map items to its average deviation
    // deviation
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

    // cache
    var preds = Map[(Int, Int), Double]()

    (u,i) => {
      var pred = preds.getOrElse((u,i), 0.0)
      if (pred == 0.0) {
        // user u average rating or global average if not in map
        val userAvg = usersAvg.getOrElse(u, globalAvg)
        // item i average deviation or 0.0 if not in map
        val itemAvgDev = itemsAvgDev.getOrElse(i, 0.0)
        // baseline prediction
        pred = (userAvg+itemAvgDev*scale((userAvg+itemAvgDev), userAvg))
        // update cache
        preds = preds+((u,i)->pred)
      }
      // return prediction
      pred
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
    
    // initiate cache
    var similarities = Map[(Int, Int), Double]()
    (u,v) => { 
      var sim = similarities.getOrElse((u,v), 0.0)
      if (sim == 0.0) {
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
        sim = ratedByBoth.toSeq.map(
          i => (preprocessedRatings.getOrElse((u,i), 0.0), preprocessedRatings.getOrElse((v,i),0.0))).map{
            case (x,y)=> x*y}.sum
        // update cache
        similarities = similarities+((u,v)->sim)
      }
      // return similarity value
      sim
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

    // initiate cache
    var coefficients = Map[(Int, Int), Double]()
    (u,v) => {
      var coeff = coefficients.getOrElse((u,v), 0.0)
      if(coeff == 0.0) {
        // get sequence of ratings for each user u and v
        val uRatings = ratedByUs.getOrElse(u, Nil)
        val vRatings = ratedByUs.getOrElse(v, Nil)

        // items rated by both users
        val ratedByBoth = uRatings.map(_.item).toSet.intersect(vRatings.map(_.item).toSet)
        // number of items rated by both users
        val sizeIntersection = ratedByBoth.size 
        // compute jaccard coefficient
        coeff = sizeIntersection/(uRatings.size+vRatings.size-sizeIntersection)
        // update cache
        coefficients = coefficients+((u,v)-> coeff)
      }
      coeff
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

    // initiate cache
    var sums = Map[(Int, Int), Double] ()

    (u, i) => {
      var sum = sums.getOrElse((u,i), 0.0)
      if (sum == 0.0) {
        // get i's ratings
        val ratedIUsers = ratedI.getOrElse(i, Nil)
        // get similarity values between user u and all others that rated i
        var ss = ratedIUsers.map(x=>x.user-> similarityFunction(x.user, u)).toMap
        // compute denominator of weighted sum
        var ssSum = ss.mapValues(_.abs).values.sum
        // weighted sum deviation
        if (ssSum!=0.0){
          sum = ratedIUsers.map(
            x=> ss.getOrElse(x.user,0.0)*normalizedDeviations.getOrElse((x.user, i), 0.0)).sum / ssSum
        } else sum = 0.0
        // update cache
        sums = sums+((u,i)-> sum)
      }
      // return sum
      sum
    }
  }

  /**
  * Personnalized Predictor predicting for (user, item) the baseline prediction 
  *   and using the user-specific weighted-sum deviation as deviation
  * @param ratings a sequence of ratings
  * @return function that maps (user, item) to its prediction
  */
  def predictor(ratings : Seq[Rating], wsd : (Int, Int) => Double) : (Int, Int)=>Double = {
    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // map each user to its average rating
    val usersAvgValue = usersAvg(ratings)
    // // map each user to its weighted-sum deviation
    // val wsd = weightedSumDeviation(ratings, similarityFunction)

    // initiate cache
    var preds = Map[(Int, Int), Double]()

    (u,i) => {
      var pred = preds.getOrElse((u,i), 0.0)
      if (pred == 0.0) {
        // u average rating
        var userAvg = usersAvgValue.getOrElse(u, globalAvgValue) 
        // u weighted-sum deviation
        var userWSD = wsd(u, i)
        // prediction
        pred = userAvg+userWSD*scale((userAvg+userWSD), userAvg)
        // update cache
        preds = preds+((u,i)->pred)
      }
      // return prediction
      pred
    } 
  }



 /**
  * Compute all the neighbors of a user and their similarities
  * @param ratings a sequence of ratings
  * @param k number of neighbors to keep
  * @return function that map user to sequence of neighbors and their similarity score
  */
  def getNeighbors(ratings : Seq[Rating], k : Int) : Int => Seq[(Int, Double)] = {
  
    // defined adjusted cosine similairy function 
    val similarityFunction = adjustedCosineSimilarityFunction(ratings)

    // set of users in ratings dataset
    val allUsers = ratings.map(x=>x.user).toSet
    // initiation of neighbors map
    var allNeighbors = Map[Int, Seq[(Int, Double)]]()

    u => {
      // get all neighbors of u if already computed
      var uNeighbors = allNeighbors.getOrElse(u, Nil)
      if (uNeighbors.isEmpty) { // neighbors of u never computed
        // all possible neighbors of u
        val others = (allUsers- u).toSeq
        // compute similarity between u and possible neighbors and choose k most similar
        uNeighbors = others.map(x=>(x, similarityFunction(u,x))).sortWith(_._2>_._2).take(k)
        // update neighbors map
        allNeighbors = allNeighbors+(u-> uNeighbors)
      }
      // u's neighbors along with their similarity score
      uNeighbors
    }
  } 

  /**
  * Compute the similarity of pair of users taking into consideration neighborhood
  * @param ratings a sequence of ratings
  * @param k number of neighbors to consider
  * @param sim similairy function to user
  * @return function that map (user1, user2) to their similarity score
  */
  def getSimilarity (ratings : Seq[Rating], k : Int, sim: (Int, Int)=> Double) : (Int, Int) => Double = {
    // map each user to its neighborhood
    val nn = getNeighbors(ratings, k)

    // initiate cache
    var similarities = Map[(Int, Int), Double]()

    // compute similarity between 2 users
    (user1: Int, user2: Int) => {
      var sim = similarities.getOrElse((user1, user2), 0.0)
      if (sim == 0.0) {
        // similarity
        var sim = nn(user1).map(x=>{
        if (x._1==user2){x._2}
        else {0.0}
        }).sum

        // update cache
        similarities = similarities+((user1, user2)-> sim)
      }
      // return similarity
      sim
    }
  }

  /**
  * Compute user-specific weighted-sum deviation function taking into consideration its neighborhood
  * @param ratings a sequence of ratings
  * @param k number of nieghbors to keep in neighborhood
  * @return map each (user, item) to its weighted-sum deviation
  */
  def weightedSumDevKNN (ratings : Seq[Rating], k : Int, simFunction : (Int, Int) => Double) : (Int, Int) => Double = {

    // global average rating
    val globalAvgValue = globalAvg(ratings)
    // find k-nearest neighbors per user
    val allNeighbors = getNeighbors(ratings, k)
    
    // define similarity function
    // val simFunction = getSimilarity(ratings, k, adjustedCosineSimilarityFunction(ratings))
    // map each user to its average rating
    val usersAvgValue = usersAvg(ratings)

    // map each item to sequence of ratings
    var ratingsByItems = ratings.groupBy(_.item)

    // initiate cache
    var sums = Map[(Int, Int), Double]()

    (u : Int,i : Int) => {
      var sum = sums.getOrElse((u,i), 0.0)
      if (sum == 0.0) {
        // sequence of ratings that rated i
        val users = ratingsByItems.getOrElse(i, Seq[Rating]())

        // user u average rating
        val userAvg = usersAvgValue.getOrElse(u, globalAvgValue)

        // u's neighborhood
        val neighbors = allNeighbors(u).toMap

        // compute for each ratings that rated i the similarity score with u and the normalized deviation
        val simVal = users.map(x=> {
          // get x's user average rating
          val avgU = usersAvgValue.getOrElse(x.user, globalAvgValue)
          ((x.rating-avgU)/scale(x.rating, avgU), simFunction(u, x.user))
        })

        // compute numerator and denominator of weighted-sum deviation
        val ssSum = simVal.foldLeft((0.0,0.0)) {
          (acc, a) => {
            (acc._1+a._1*a._2, acc._2+a._2.abs)
          }
        }

        // user u weighted-sum deviation
        if (ssSum._2 >0) {
          sum = ssSum._1/ssSum._2
        } else sum = 0.0

        // update cache
        sums = sums + ((u,i)-> sum)
      }
      // return user u weighted-sum deviation
      sum 
    }
    }


  def recommendations (ratings : Seq[Rating], predictor : (Int, Int) => Double) : (Int, Int) => Seq[(Int, Double)] = {
    // val predictor = predictor(train, weightedSumDevKNN(ratings, k, sim))

    val order = (x:(Int, Double), y:(Int, Double)) => {
      if (x._2 == y._2) {
        x._1<y._1
      } else {
        x._2>y._2
      }
    }

    var recommend = Map[(Int, Int), Seq[(Int, Double)]]()

    (user : Int, n : Int) => {
      var reco = recommend.getOrElse((user,n), Seq[(Int, Double)]())
      if (reco.isEmpty) {
        val notRated = ratings.map(_.item).toSet.diff(ratings.filter(x=> x.user == user).map(_.item).toSet)
        reco = notRated.toSeq.map(x=> (x, predictor(user,x))).sortWith(order).take(n)
        recommend = recommend + ((user,n) -> reco)
      }
      reco
    }

  }

}

