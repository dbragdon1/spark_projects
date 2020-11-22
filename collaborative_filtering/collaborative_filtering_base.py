from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import IntegerType
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from numpy import array
import sys
sys.path.append('..')
print(sys.path)
from pyspark.ml import Pipeline
#from .helper_functions import load_amazon_cellphones
from helper_functions import load_amazon_cellphones


if len(sys.argv) == 2:
    num_train_samples = int(sys.argv[1])
else:
    num_train_samples = -1

data = load_amazon_cellphones(num_train_samples)

rel_data = map(lambda x: (x['reviewerID'], 
                                x['asin'], 
                                x['overall']), 
                          data)

spark = SparkSession.builder.master("local[*]") \
                            .config('spark.driver.memory', '15g') \
                            .appName('Collaborative_Filtering').getOrCreate()

sc = spark.sparkContext

data_df = sc.parallelize(rel_data).toDF(schema = ['userID', 'productID', 'rating'])

#Converting user hashes to unique labels
userIndexer = StringIndexer(inputCol = 'userID', outputCol = 'userIDLabel')

#Converting product hashes to unique labels
productIndexer = StringIndexer(inputCol = 'productID', outputCol = 'productIDLabel')

pipeline = Pipeline(stages = [userIndexer, productIndexer])

pipelineFit = pipeline.fit(data_df)
data_df = pipelineFit.transform(data_df)

data_df = data_df.withColumn('userIDLabel', data_df['userIDLabel'].cast(IntegerType()))
data_df = data_df.withColumn('productIDLabel', data_df['productIDLabel'].cast(IntegerType()))
data_df = data_df.select('userIDLabel', 'productIDLabel', 'rating')

als = ALS(userCol = 'userIDLabel', 
          itemCol = 'productIDLabel', 
          ratingCol = 'rating',
          coldStartStrategy = 'drop')

train_data, test_data = data_df.randomSplit([0.9, 0.1])
model = als.fit(train_data)

evaluator = RegressionEvaluator(metricName = "rmse", 
                                labelCol = "rating", 
                                predictionCol = "prediction")

train_predictions = model.transform(train_data)
train_rmse = evaluator.evaluate(train_predictions)

test_predictions = model.transform(test_data)
test_rmse = evaluator.evaluate(test_predictions)

print(train_rmse, test_rmse)