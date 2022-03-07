package test.predict

import org.scalatest._
import funsuite._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import tests.shared.helpers._
import ujson._

// Added import 
import predict.Baseline._

class BaselineTests extends AnyFunSuite with BeforeAndAfterAll {

   val separator = "\t"
   var spark : org.apache.spark.sql.SparkSession = _

   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : Array[shared.predictions.Rating] = null
   var test2 : Array[shared.predictions.Rating] = null

   override def beforeAll {
       Logger.getLogger("org").setLevel(Level.OFF)
       Logger.getLogger("akka").setLevel(Level.OFF)
       spark = SparkSession.builder()
           .master("local[1]")
           .getOrCreate()
       spark.sparkContext.setLogLevel("ERROR")

       // For these questions, train and test are collected in a scala Array
       // to not depend on Spark
       train2 = load(spark, train2Path, separator).collect()
       test2 = load(spark, test2Path, separator).collect()
   }

   // All the functions definitions for the tests below (and the tests in other suites) 
   // should be in a single library, 'src/main/scala/shared/predictions.scala'.

   // Provide tests to show how to call your code to do the following tasks (each in with their own test):
   // each method should be invoked with a single function call. 
   // Ensure you use the same function calls to produce the JSON outputs in
   // src/main/scala/predict/Baseline.scala.
   // Add assertions with the answer you expect from your code, up to the 4th
   // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
   test("Compute global average")                           { assert(within(globalAvg(train2), 3.5264, 3.5264+ 0.0001)) }
   test("Compute user 1 average")                           { assert(within(userAvgPredictor(train2)(1,0), 3.6330, 3.6330+0.0001)) }
   test("Compute item 1 average")                           { assert(within(itemAvgPredictor(train2)(0,1), 3.8882, 3.8882+0.0001)) }
   test("Compute item 1 average deviation")                 { assert(within(computeAllDevs(train2, computeAllUsersAvg(train2))(1), 0.3027, 0.3027+0.0001)) }
   test("Compute baseline prediction for user 1 on item 1") { assert(within(formulaPredictor(train2)(1,1), 4.0468, 4.0468+0.0001)) }

   // Show how to compute the MAE on all four non-personalized methods:
   // 1. There should be four different functions, one for each method, to create a predictor
   // with the following signature: ````predictor: (train: Seq[shared.predictions.Rating]) => ((u: Int, i: Int) => Double)````;
   // 2. There should be a single reusable function to compute the MAE on the test set, given a predictor;
   // 3. There should be invocations of both to show they work on the following datasets.
   test("MAE on all four non-personalized methods on data/ml-100k/u2.base and data/ml-100k/u2.test") {
     assert(within(MAE(test2, globalAvgPredictor(train2)), 0.9489, 0.9489+0.0001))
     assert(within(MAE(test2, userAvgPredictor(train2)), 0.8383, 0.8383+0.0001))
     assert(within(MAE(test2, itemAvgPredictor(train2)), 0.8206, 0.8206+0.0001))
     assert(within(MAE(test2, formulaPredictor(train2)), 0.7604, 0.7604+0.0001))
   }
}
