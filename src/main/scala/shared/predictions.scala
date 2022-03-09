package shared
import org.apache.spark.rdd.RDD
import scala.math

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
  * Baseline related functions
  */

  /**
  * Compute the MAE on the given data set 
  * @param data the data set on which to compute the MAE
  * @param predict function used to make a prediction for the rating
  * @return the MAE
  */
  def MAE(data: Seq[Rating], predict: (Int, Int)=> Double): Double = {
    applyAndMean(data){
      x => (x.rating-predict(x.user, x.item)).abs
    }
  }

  /** Compute every component to produce rating based on the global average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def globalAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    val globalAvgValue = globalAvg(train)
    (user, item) => globalAvgValue
  }

  /** Compute every component to produce rating based on the user average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def userAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    lazy val globalAvgValue = globalAvg(train)
    val usersAvg = computeAllUsersAvg(train)
    (user, item) => usersAvg.getOrElse(user, globalAvgValue)
  }

  /** Compute every component to produce rating based on the item average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def itemAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    lazy val globalAvgValue = globalAvg(train)
    val itemsAvg = computeAllItemsAvg(train)

    (user, item) => itemsAvg.getOrElse(item, globalAvgValue)
  }

  /** Compute every component to produce rating based on the formula in the handout
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def formulaPredictor(train: Seq[Rating]): (Int, Int) => Double = {
     // Computed only if needed
    lazy val globalAvgValue = globalAvg(train)

    val usersAvg = computeAllUsersAvg(train)
    val devs = computeAllDevs(train, usersAvg)

    (user, item) => {
      val dev = devs.getOrElse(item, 0.0)
      val avg = usersAvg.getOrElse(user, globalAvgValue)
      avg + dev*scale(dev + avg, avg)
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

  /** Compute the deviation with the formula given in the handout
  *  @param data the training data set
  *  @param usAvg map between user and his.her average rating in the training data set 
  *  @return map between items and their deviance
  */
  def computeAllDevs(data: Seq[Rating], usAvg: Map[Int,Double]): Map[Int, Double] = groupByApplyMean(data, x=>x.item){
    x => {
        val avg = usAvg(x.user)
        (x.rating-avg)/scale(x.rating, avg)
    }
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


  /** 
  * Spark related prediction functions
  */

  /** 
  * Apply a function to every element in the data set an then compute the average
  * @param data the data on which to compute the average
  * @return the average value over the data set 
  */
  def applyAndAverage(data: RDD[Rating])(f: (Rating => Double)): Double = {
    val acc = data.map(x => (f(x), 1)).reduce( (x,y) => (x._1 + y._1, x._2 + y._2))
    acc._1/acc._2
  }

  /** 
  * Compute every component to produce rating based on the formula in the handout
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def predictorFunctionSpark(data: RDD[Rating]):(Int, Int )=> Double = {
    val usersAvg = computeAllUsersAvgSpark(data)
    val globalAvgValue = computeGlobalAvgSpark(data)
    val devs = computeItemDevsSpark(data, usersAvg)
    (user, item)=>{
      val dev = devs.getOrElse(item, 0.0)
      val avg = usersAvg.getOrElse(user, globalAvgValue)
      avg + dev*scale(dev + avg, avg)
    }
  }

  /** Compute every component to produce rating based on the user average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def predictorUserAvgSpark(data: RDD[Rating]): (Int, Int )=> Double = {
    val usersAvg = computeAllUsersAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => usersAvg.getOrElse(user, globalAvgValue)
  }

  /** Compute every component to produce rating based on the item average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def predictorItemAvgSpark(data: RDD[Rating]): (Int, Int)=> Double = {
    val itemsAvg = computeItemAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => itemsAvg.getOrElse(item, globalAvgValue)
  }

  /** Compute every component to produce rating based on the global average
  *  @param train the data set to use to make predictions 
  *  @return Function mapping an user and item to a rating
  */
  def predictorGlobalAvgSpark(data: RDD[Rating]): (Int, Int) => Double = {
    val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => globalAvgValue
  }

  /**
  * Compute the MAE on the given data set 
  * @param data the data set on which to compute the MAE
  * @param predict function used to make a prediction for the rating
  * @return the MAE
  */
  def MAESpark(test: RDD[Rating], predict: (Int, Int)=> Double): Double = applyAndAverage(test){x=> (x.rating-predict(x.user, x.item)).abs}

  /** Compute the total average over the whole dataset
  *  @param data the data set
  *  @return global average over the data set
  */
  def computeGlobalAvgSpark(data: RDD[Rating]): Double = applyAndAverage(data)(_.rating)

  /** Compute the average rating for each user
  *  @param data the data set
  *  @return map between user to her.his average rating
  */
  def computeAllUsersAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.user).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap


  /** Compute the average rating for each item
  *  @param data the data set
  *  @return map between item to its average rating
  */
  def computeItemAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap

  /** Compute the deviation with the formula given in the handout
  *  @param data the training data set
  *  @param usAvg map between user and his.her average rating in the training data set 
  *  @return map between items and their deviance
  */
  def computeItemDevsSpark(data: RDD[Rating], usAvg: Map[Int, Double]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq){x=>
      val avg = usAvg(x.user)
      (x.rating-avg)/scale(x.rating, avg)
    }
  }).collect().toMap



  /**
  * Personalized related functions
  */
  /**
  * Predictor return similarity one between any two users
  * @param user1 the first user 
  * @param user2 the second user
  * @return the similarity between the two users which is always 1
  */
  def simOnes = (user1: Int, user2: Int) => 1.0

  /**
  * Return a function taking one user and one item as input and output the predicted rating
  * @param train the training test to compute the predictions
  * @param sim Function outputing the similarity between two user
  * @return Function taking one user and one item as input and output the predicted rating
  *
  */
  def predictor(train: Seq[Rating])(sim: ((Int, Int)=> Double)): (Int, Int)=> Double = {
    // Compute mandatory values for the predictions
    val ratingsByItems = ratingByItems(train)
    val usAvgs = computeAllUsersAvg(train)
    val globalAvgValue = globalAvg(train)
    
    (user: Int, item: Int)=> {
      // user that rated this item 
      val users = ratingsByItems.getOrElse(item, Nil)

      // get average rating value for this user or the global average if the user didn't rate anything in the train set
      val avg = usAvgs.getOrElse(user, globalAvgValue)

      // get similarity with current user and the rating 
      val simVal = users.map(x=> {
        val avgU = usAvgs.getOrElse(x.user, globalAvgValue)
        ((x.rating-avgU)/scale(x.rating, avgU), sim(user, x.user))
      })

      // Compute the denominator as well as the numerator of the similarity between the two useres 
      val sumSim = simVal.foldLeft((0.0, 0.0)){
        (acc, x)=>{
          (acc._1 + x._1*x._2, acc._2 + x._2.abs)
        }
      }
      // compute the prediction
      val denomSum = if (sumSim._2!=0) sumSim._1/sumSim._2 else 0.0
      avg + denomSum* scale(avg+ denomSum, avg)
    }
  }

  /**
  * Group the input sequence by the given key
  * @param data the data set to group by
  * @param key the key on which to group the value
  * @return A map between the key and its group by value 
  * 
  */
  def groupBy(data: Seq[Rating])(key: Rating => Int): Map[Int, Seq[Rating]] = 
    data.foldLeft(Map[Int, Seq[Rating]]()){
      // The accumulator is a map mapping the key (an int) to a sequence of rating corresponding to all the groupby matching the key vlaue 
      (acc, x)=>{
        // We access the value already stored or get an empty list if no value was stored for this key
        val cur:Seq[Rating] = acc.getOrElse(key(x), Seq[Rating]())
        // Update of the map
        acc + (key(x) -> (x+:cur))
      }
    }

  /**
  * Get a map mapping every user to all the rating (s)he has made
  * @param data the data we want to group by
  * @return map mapping every user to all the rating (s)he has made
  */
  def ratingByUsers(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.user)
  
  /**
  * Get a map mapping every item to all the ratings made by users
  * @param data the data we want to group by
  * @return map mapping every item to all the ratings made by users
  */
  def ratingByItems(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.item)

  /**
  * Return a function mapping a pair of user to their similarity according to the Jaccard index
  * @param data the data we want to group by
  * @return a function mapping a pair of user to their similarity according to the Jaccard index
  */
  def jaccardSimCoef(data: Seq[Rating]): (Int, Int )=> Double = {
    val ratUsers = ratingByUsers(data)
    // Map that will be used to cache the value so that we repeated access can be speed up
    var cache = Map[(Int, Int), Double]()

    (user1, user2)=>{
      // Look into the cache if the similarity has already been computed
      val sim = cache.getOrElse((user1, user2), -1)
      if(sim < 0){
        // If the similarity hasn't been computed yet, get all the items rated by both users
        val items1 = ratUsers.getOrElse(user1, Nil).map(_.item)
        val items2 = ratUsers.getOrElse(user2, Nil).map(_.item)

        
        val  sim2 = {
          if(items1.length==0 || items2.length==0){
            // If the length of one list is zero, they have no item in common
            0.0
          }else{
            // Compute length
            val set1 = items1.toSet
            val set2 = item2.toSet
            val inter_len = set1.intersect(set2).length
            val size1 = set1.length
            val size2 = set2.length 
            // Jaccard index |intersection|/|union|
            inter_len/(size1+size2-inter_len)
          }
        }
        // Update the cache 
        cache = (cache +((user1, user2)->sim2)) + (user2, user1) -> sim2)
        // Output the similarity
        sim2 
      }else{
        sim
      }
    }
  }

  /**
  * Return a function computing the similarity according to the adjusted cosine similarity
  * @param train the training data
  * @return function computing the similarity according to the adjusted cosine similarity
  */
  def adjustedCosine(train:Seq[Rating]): (Int, Int)=> Double = {
    // Compute mandatory values
    val usAvg = computeAllUsersAvg(train)
    lazy val globalAvgValue = globalAvg(train)

    // Function computing the deviance according to the handout formula
    val dev = (x: Rating) => {
      val avg = usAvg.getOrElse(x.user, globalAvgValue)
      (x.rating-avg)/scale(x.rating, avg)
    }

    // map every rating to the (user, deviance)
    val mapped = train.map(x=>(x.user, dev(x)))

    // Compute the norm (denominator) for each user in the dataset
    val normByUsers = mapped.foldLeft(Map[Int,Double]()){
      // The accumulator is a map mapping a user Id to the sum of square rating
      (acc,x) =>
       val cur:Double = acc.getOrElse(x._1, 0.0)
        // Update of the map
        acc + (x._1 -> (cur+x._2*x._2))
    }.mapValues{
      // Perform the square root to get the norm
      x=> math.sqrt(x)
    }

    // Get all the map of all ratings by user
    val itemsByU = ratingByUsers(train)

    // This map will be used as cache to avoid double computation
    var cache = Map[(Int, Int), Double]()

    // The returned function
    (user1: Int, user2: Int)=> {
      // 
      val sim:Double = cache.getOrElse((user1,user2),-1.0)
      if(sim<0){
        // Get all the ratings for each user 
        val ratings1 = itemsByU.getOrElse(user1, Nil)
        val ratings2 = itemsByU.getOrElse(user2, Nil)

        if(ratings1.length==0 || ratings2.length==0){
          // if one of the user didn't rate anything we output a similarity of 0
          0.0
        }else{
          //Get the set of item rated by each user and compute intersection
          val items1 = ratings1.map(_.item)
          val items2 = ratings2.map(_.item)
          val inter = items1.toSet.intersect(items2.toSet)

          // Get the pre-computed norm for each user or 0 if the user wasn't in the training set 
          val norm2 = normByUsers.getOrElse(user2, 0.0)
          val norm1 = normByUsers.getOrElse(user1, 0.0)

          val remaining2 = {
            if(norm2 == 0.0){
              // if the norm is 0 we simply return an empty as we will not use the rating of this user
              Map[Int, Double]()
            }else{
            ratings2.foldLeft(Map[Int, Double]()){
              (acc, x)=>{
                // If the item is in the intersection, map the item to the ratio deviation over norm 
                if(inter.contains(x.item)){ acc + (x.item -> (dev(x)/norm2))}
                else acc
              }
            }
          }
          
          // Compute the similarity according to the handout formula 
          val sim:Double = {
            if(norm1 == 0){
              // if the norm is 0  we return 0 as this means that we have no ratings
              0.0
            }else{
            ratings1.foldLeft(0.0){
                (acc, x)=>{
                  if(inter.contains(x.item)){
                    // if the item is rated by both user we compute the weighted sum
                    acc + dev(x) / norm1 *remaining2(x.item)
                    //Otherwise we ignore the item.
                  } else acc
                }
              }
            }
          }
          // Cache the computed similarity
          cache = (cache +((user1, user2)->sim)) + ((user2, user1)->sim)
          sim
        }
      }else{
        sim
      }
    }
  }

  /**
  *
  * KNN related functions 
  *
  */

  /**
  * Get the similarity of two users based on k-nearest neighbours
  * @param train the training set on which to fit the predictor
  * @param k the number of neighbors we want to take into account
  * @param sim function measuring the similarity between two users
  * @return a function returning the similarity between two users in the knn sense
  */
  def getSimilarity(train: Seq[Rating], k: Int, sim: (Int, Int)=> Double): (Int, Int)=> Double = {
    val nn = kNearestNeighbours(train, k, sim)
    (user1: Int, user2: Int) => {
      nn(user1).map(x=>{
        if (x._1==user2){x._2}
        else {0.0}
        }).sum
    }
  }

  /**
  * Return the list of the k nearest neighbours as well as their similarity for the given user
  * @param train the training set on which to fit the predictor
  * @param k the number of neighbors we want to take into account
  * @param sim function measuring the similarity between two users
  * @return function mapping a user id to the list of k (neigbourdid, similarity) pairs 
  */
  def kNearestNeighbours(train: Seq[Rating], k: Int,  sim: (Int, Int)=> Double): Int => List[(Int, Double)] = {
    val allUsers = train.map(x=>x.user).toSet
    var map = Map[Int, List[(Int, Double)]]() // map that will act as a local cache
    (user: Int )=>{
      var neighbours = map.getOrElse(user, Nil)
      if (neighbours.length == 0){
        val allOthers = (allUsers-user).toList
        neighbours = allOthers.map(x=> (x, sim(user, x))).sortBy(_._2)(Ordering[Double].reverse).take(k)
        map = map +(user-> neighbours)
      }
      neighbours
    }
  }

  /**
  * Recommender related functions
  *
  */

}
