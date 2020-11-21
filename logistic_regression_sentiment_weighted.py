from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pyspark.sql.functions as F
from itertools import chain
import re
from nltk.corpus import stopwords
from string import punctuation
import nltk
import sys
from helper_functions import get_class_weights, load_amazon_cellphones

nltk.download('stopwords')
sw = list(set(stopwords.words('english')))

if len(sys.argv) == 2:
  num_train_samples = int(sys.argv[1])
else:
  num_train_samples = -1

print('Loading raw data file.')

lines = load_amazon_cellphones(num_train_samples)

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

print('Calculating Class Weights. \n')
data = get_class_weights(data, labelCol = 'label')

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

lr = LogisticRegression(featuresCol = 'features', 
                        labelCol = 'label',
                        weightCol = 'weight')

pipeline = Pipeline(stages = [regexTokenizer, stopwordsRemover, countVectorizer, lr])

print('Splitting data.\n')
train_data, test_data = data.randomSplit([0.9, 0.1])

print('Fitting training data to pipeline.\n')
pipelineFit = pipeline.fit(train_data)
transformed_train_data = pipelineFit.transform(train_data)

print('Predicting on testing data.\n')
test_predictions = pipelineFit.transform(test_data)

print('Calculating Metrics.\n')
metrics = BinaryClassificationMetrics(test_predictions.select('prediction', 'label').rdd)

auroc = metrics.areaUnderROC
aupr = metrics.areaUnderPR

print('Area under ROC: {}'.format(auroc))
print('Area under Precision/Recall Curve: {}'.format(aupr))

with open('metrics/logistic_regression_sentiment_results.txt', 'w') as metricfile:
    num_samples = train_data.count()
    metricfile.write('Number of training samples after undersampling: {} \n'.format(num_samples))
    metricfile.write('Area under ROC: {} \n'.format(auroc))
    metricfile.write('Area under Precision/Recall Curve: {} \n'.format(aupr))

#modelpath = '/models/logistic_regression_sentiment/'
#pipelineFit.save(sc, modelpath)
#pipelineFit.write().overwrite().save(modelpath)
#lrModel.write().overwrite().save(modelpath)