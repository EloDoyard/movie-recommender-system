package shared

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

  def MAE(data: Seq[Rating], predict: (Int, Int)=> Double): Double = {
    applyAndMean(data){
      x => (x.rating-predict(x.user, x.item)).abs
    }
  }

  def globalAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    val globalAvgValue = globalAvg(train)
    (user, item) => globalAvgValue
  }

  def userAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    lazy val globalAvgValue = globalAvg(train)
    val usersAvg = computeAllUsersAvg(train)
    (user, item) => userAvg(usersAvg, user, globalAvgValue)
  }

  def itemAvgPredictor(train: Seq[Rating]): (Int, Int) => Double = {
    lazy val globalAvgValue = globalAvg(train)
    val itemsAvg = computeAllItemsAvg(train)

    (user, item) => itemAvg(itemsAvg, item, globalAvgValue)
  }

  def formulaPredictor(train: Seq[Rating]): (Int, Int) => Double = {
     // Computed only if needed
    lazy val globalAvgValue = globalAvg(train)

    val usersAvg = computeAllUsersAvg(train)
    val devs = computeAllDevs(train, usersAvg)

    (user, item) => predict(devs, user, item, usersAvg, globalAvgValue)
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
  def computeAllDevs(data: Seq[Rating], usAvg: Map[Int,Double]): Map[Int, Double] = groupByApplyMean(data, x=>x.item){
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


  /** 
  * Spark related prediction functions
  */

  def applyAndAverage(data: RDD[Rating])(f: (Rating => Double)): Double = {
    val acc = data.map(x => (f(x), 1)).reduce( (x,y) => (x._1 + y._1, x._2 + y._2))
    acc._1/acc._2
  }

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

  def predictorUserAvgSpark(data: RDD[Rating]): (Int, Int )=> Double = {
    val usersAvg = computeAllUsersAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => usersAvg.getOrElse(user, globalAvgValue)
  }

  def predictorItemAvgSpark(data: RDD[Rating]): (Int, Int)=> Double = {
    val itemsAvg = computeItemAvgSpark(data)
    lazy val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => itemsAvg.getOrElse(item, globalAvgValue)
  }

  def predictorGlobalAvgSpark(data: RDD[Rating]): (Int, Int) => Double = {
    val globalAvgValue = computeGlobalAvgSpark(data)
    (user, item) => globalAvgValue
  }

  def MAE(test: RDD[Rating], predict: (Int, Int)=> Double): Double = applyAndAverage(test){x=> (x.rating-predict(x.user, x.item)).abs}

  def computeGlobalAvgSpark(data: RDD[Rating]): Double = applyAndAverage(data)(_.rating)

  def computeAllUsersAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.user).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap

  def computeItemAvgSpark(data: RDD[Rating]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq)(_.rating)
  }).collect().toMap

  def computeItemDevsSpark(data: RDD[Rating], usAvg: Map[Int, Double]): Map[Int, Double] = data.groupBy(_.item).mapValues(y=>{
    applyAndMean(y.toSeq){x=>
      val avg = usAvg(x.user)
      (x.rating-avg)/scale(x.rating, avg)
    }
  }).collect().toMap



  /**
  * Personalized related functions
  */
  
  def MAE(data: Seq[Rating], predict: (Int, Int)=> Double): Double = {
    val acc = data.foldLeft((0.0,0)){
      (acc, x)=> {
        (acc._1+(predict(x.user, x.item)-x.rating).abs, acc._2+1)
      }
    }
    acc._1/acc._2
  }

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
    println(mapped)

    val normByUsers = mapped.foldLeft(Map[Int,Double]()){(acc,x) =>
       val cur:Double = acc.getOrElse(x._1, 0.0)
        // Update of the map
        acc + (x._1 -> (cur+x._2*x._2))
    }

    val itemsByU = ratingByUsers(train)
    println(itemsByU)
    println(normByUsers)

    (user1: Int, user2: Int)=> {
      val ratings1 = itemsByU.getOrElse(user1, Nil)
      val ratings2 = itemsByU.getOrElse(user2, Nil)
      if(ratings1.length==0 || ratings2.length==0) 0.0
      else{
        val items1 = ratings1.map(_.item)
        val items2 = ratings2.map(_.item)
        val inter = items1.toSet.intersect(items2.toSet)
        println(inter)

        val remaining2 = ratings2.foldLeft(Map[Int, Double]()){
          (acc, x)=>{
            if(inter.contains(x.item)){ acc + (x.item -> (dev(x)/normByUsers(user2)))}
            else acc
          }
        }
        println(remaining2)

        ratings1.foldLeft(0.0){
          (acc, x)=>{
            if(inter.contains(x.item)){
              val norm = normByUsers.getOrElse(user1, 0.0)
              if(norm==0) acc
              else acc + dev(x) / norm *remaining2(x.item)
            } else acc
          }
        }
      }
    }
  }

}
