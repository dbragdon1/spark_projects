from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import col
import re
from nltk.corpus import stopwords
import json
from string import punctuation
import nltk
import os 
import sys

num_train_samples = int(sys.argv[1])

nltk.download('stopwords')
sw = list(set(stopwords.words('english')))

print('Loading raw data file.')
rawfile = open("data/cellphones.json", 'r')
#load each line as separate json object
lines = []
for i, line in enumerate(rawfile):
  lines.append(json.loads(line))
  if i == num_train_samples - 1:
    break

print('Building Spark Session.')
spark = SparkSession.builder.master("local[*]") \
                            .config('spark.driver.memory', '15g') \
                            .appName('amazon_phones').getOrCreate()
sc = spark.sparkContext

print("Example row: ")
print(lines[0])

#---Helper functions---

#Maps reviews to positive/negative
def create_categories(x):
  if x >= 4.0:
    return 1.0
  else:
    return 0.0

#function for preprocessing data
def clean(review):
  #removing numbers
  review = re.sub('[0-9]', '', review).lower()
  #removing punctuation
  review = review.translate(str.maketrans('', '', punctuation))
  return review


print('Preparing Data.\n')
relevant_info = [(line['reviewText'], 
                 line['overall']) 
                 for line in lines]

print(len(relevant_info))

print('Parallelizing and Preprocessing.\n')
rdd = sc.parallelize(relevant_info)


#Removing Neutral reviews
rdd = rdd.filter(lambda x: x[1] != 3.0)

#converting scores into integer
rdd = rdd.map(lambda x: (x[0], create_categories(x[1])))

#preprocessing reviews
rdd = rdd.map(lambda x: (clean(x[0]), x[1]))

#removing empty reviews
rdd = rdd.filter(lambda x: x[0] != '')

#Convert RDD to DataFrame object
print('Converting RDD to DataFrame.')
schema = ['review', 'label']
data = rdd.toDF(schema = schema)
data.show()
data.printSchema()
#data.select('label').describe().show()

#Undersampling positive reviews to avoid class imbalance
major_df = data.filter(col('label') == 1)
minor_df = data.filter(col('label') == 0)

sampled_majority_df = major_df.sample(False, .16)
undersampled_data = sampled_majority_df.unionAll(minor_df)
#undersampled_data.groupBy('label').count().show()

#Begin building preprocessing pipeline

#Steps: Word tokenize --> remove stopwords --> convert to BoW vectors
print('Building Pipeline.\n')
regexTokenizer = RegexTokenizer(inputCol = 'review',
                                outputCol = 'tokenized',
                                pattern = "\\W")

stopwordsRemover = StopWordsRemover(inputCol = 'tokenized',
                                    outputCol = 'removed_sw').setStopWords(sw)

countVectorizer = CountVectorizer(inputCol = "removed_sw",
                               outputCol = "features", 
                               vocabSize = 10000, 
                               minDF = 5)

pipeline = Pipeline(stages = [regexTokenizer, stopwordsRemover, countVectorizer])

#Split data into training and testing partitions
print('Splitting data.\n')
train_data, test_data = undersampled_data.randomSplit([0.9, 0.1])

#Fit and transform training data
print('Fitting training data to pipeline.\n')
pipelineFit = pipeline.fit(train_data)
transformed_train_data = pipelineFit.transform(train_data)
#transformed_train_data.show()

#Begin Logistic Regression
print('Training Logistic Regression Model.\n')
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')
lrModel = lr.fit(transformed_train_data)

#Predict on test data
print('Predicting on testing data.\n')
test_predictions = lrModel.transform(pipelineFit.transform(test_data))

#Calculate Metrics
print('Calculating Metrics.\n')
metrics = BinaryClassificationMetrics(test_predictions.select('prediction', 'label').rdd)

print('Area under ROC: {}'.format(metrics.areaUnderROC))
print('Area under Precision/Recall Curve: {}'.format(metrics.areaUnderPR))




