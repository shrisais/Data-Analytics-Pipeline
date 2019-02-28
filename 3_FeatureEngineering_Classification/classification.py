import sys
import nltk
import string
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from functools import reduce 
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import NaiveBayes

# Configure Spark
conf = SparkConf().setAppName("DataPipeline")
conf = conf.setMaster("local[*]")
sc   = SparkContext(conf=conf)
spark = SparkSession(sc)
sqlContext = SQLContext(sc)


politics_df = spark.read.text('/home/hadoop/spark/Classification/Politics/*')
politics_df = politics_df.withColumn("category",lit("politics"))
business_df = spark.read.text('/home/hadoop/spark/Classification/Business/*')
business_df = business_df.withColumn("category",lit("business"))
sports_df = spark.read.text('/home/hadoop/spark/Classification/Sports/*')
sports_df = sports_df.withColumn("category",lit("sports"))
technology_df = spark.read.text('/home/hadoop/spark/Classification/Technology/*')
technology_df = technology_df.withColumn("category",lit("technology"))


#Combine all the dataframes
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

final_df = unionAll(politics_df,business_df,sports_df,technology_df)

data = final_df.select([column for column in final_df.columns])
data.show(10)


#Regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="value", outputCol="words", pattern="\\W")

#Stop words removal
add_stopwords = nltk.corpus.stopwords.words('english')
text_file = open("/home/hadoop/spark/Classification/extrawords.txt", "r")
add_extrawords =  text_file.read().split('\n')
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filter").setStopWords(add_stopwords)
extrawordsRemover = StopWordsRemover(inputCol="filter", outputCol="filtered").setStopWords(add_extrawords)

#Encoding string column of labels to column of label indices
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")


#Generates features using HashingTF-IDF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=500)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

#Fitting the pipeline to the training documents
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover,extrawordsRemover, hashingTF, idf, label_stringIdx])
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(10)


#Partioning Data into training and testing
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)


#Logistic Regression
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
lr_accuracy = str(evaluator.evaluate(predictions))


#Naive Bayes
nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)
predictions1 = model.transform(testData)
predictions1.filter(predictions1['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

evaluator1 = MulticlassClassificationEvaluator(predictionCol="prediction")
nb_accuracy = str(evaluator1.evaluate(predictions1))


#Random Forest

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# Train model with Training Data
rfModel = rf.fit(trainingData)
predictions2 = rfModel.transform(testData)
predictions2.filter(predictions2['prediction'] == 0) \
    .select("value","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

evaluator2 = MulticlassClassificationEvaluator(predictionCol="prediction")
rf_accuracy = str(evaluator2.evaluate(predictions2))



#Accuracy table
print "_______________________________________"
print "| Algorithm            |  Accuracy     |"
print "| Logistic Regression  | "+lr_accuracy+"|"
print "| Naive Bayes          | "+nb_accuracy+"|"
print "| Random Forest        | "+rf_accuracy+"|"
print "|______________________________________|"

