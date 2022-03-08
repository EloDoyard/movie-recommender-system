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

  def simOnes = (user1: Int, user2: Int) => 1.0

  def predictor(train: Seq[Rating])(sim: ((Int, Int)=> Double)): (Int, Int)=> Double = {
    // Compute necessary values for the predictions
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

  def groupBy(data:Seq[Rating])(key: Rating => Int): Map[Int, Seq[Rating]] = 
    data.foldLeft(Map[Int, Seq[Rating]]()){
      // The accumulator is a map mapping the key (an int) to a pair (Double, Int) corresponding to the running sum of the value we want to compute and the number of computed value respectively 
      (acc, x)=>{
        // We access the value already stored or get 0 if no value was stored for this key
        val cur:Seq[Rating] = acc.getOrElse(key(x), Seq[Rating]())
        // Update of the map
        acc + (key(x) -> (x+:cur))
      }
    }

  def ratingByUsers(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.user)
  
  def ratingByItems(data: Seq[Rating]): Map[Int, Seq[Rating]] = groupBy(data)(_.item)


  def jaccardSimCoef(data: Seq[Rating]): (Int, Int )=> Double = {
    val ratUsers = ratingByUsers(data)

    (user1, user2)=>{
      val items1 = ratUsers.getOrElse(user1, Nil).map(_.item)
      val items2 = ratUsers.getOrElse(user2, Nil).map(_.item)

      if(items1.length==0 || items2.length==0) 0.0
      else{
        val inter = items1.toSet.intersect(items2.toSet)
        val union = items1.toSet.union(items2.toSet)

        inter.size/union.size
      }
    }
  }

  def adjustedCosine(train:Seq[Rating]): (Int, Int)=> Double = {
    val usAvg = computeAllUsersAvg(train)
    lazy val globalAvgValue = globalAvg(train)

    // Compute the deviation according to the handout formulas
    val dev = (x: Rating) => {
      val avg = usAvg.getOrElse(x.user, globalAvgValue)
      (x.rating-avg)/scale(x.rating, avg)
    }

    val mapped = train.map(x=>(x.user, dev(x)))
    //println(mapped)

    val normByUsers = mapped.foldLeft(Map[Int,Double]()){(acc,x) =>
       val cur:Double = acc.getOrElse(x._1, 0.0)
        // Update of the map
        acc + (x._1 -> (cur+x._2*x._2))
    }.mapValues{
      x=> math.sqrt(x)
    }

    val itemsByU = ratingByUsers(train)
    //println(itemsByU)
    //println(normByUsers)

    var cache = Map[(Int, Int), Double]()

    (user1: Int, user2: Int)=> {
      val sim:Double = cache.getOrElse((user1,user2),-1.0)
      if(sim<0){
        val ratings1 = itemsByU.getOrElse(user1, Nil)
        val ratings2 = itemsByU.getOrElse(user2, Nil)
        if(ratings1.length==0 || ratings2.length==0) 0.0
        else{
          val items1 = ratings1.map(_.item)
          val items2 = ratings2.map(_.item)
          val inter = items1.toSet.intersect(items2.toSet)

          val norm2 = normByUsers.getOrElse(user2, 0.0)
          val norm1 = normByUsers.getOrElse(user1, 0.0)

          val remaining2 = ratings2.foldLeft(Map[Int, Double]()){
            (acc, x)=>{
              if(norm2==0.0) acc
              else if(inter.contains(x.item)){ acc + (x.item -> (dev(x)/norm2))}
              else acc
            }
          }
          //println(remaining2)

          val sim:Double = ratings1.foldLeft(0.0){
            (acc, x)=>{
              if(inter.contains(x.item)){
                if(norm1==0) acc
                else acc + dev(x) / norm1 *remaining2(x.item)
              } else acc
            }
          }
          cache = (cache +((user1, user2)->sim)) + ((user2, user1)->sim)
          sim
        }
      }else{
        sim
      }
    }
  }

}
