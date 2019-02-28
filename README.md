# Data Analytics Pipeline

### Environment Details:
Following are the environment chosen for this Lab:
1. Apache Spark run on Virtual Machine
2. Scripts were written in Python


### Documentation

1. Collect data   
Data was collected using NYTimesArticles-API for the following categories:
  - Business : Stock, Economy, Finance
  - Sports : NBA, BFL, Golf
  - Politics : President, Trump, Election
  - Entertainment : Met Gala, TMZ, SNL
2. Dataframe   
Read all articles from each of the category and appended it into a spark-dataframe along with a column specifying the category name.
3. Feature Engineering   
  - Tokenizer : Tokenize each article into words using space delimiter. Used RegexTokenizer API from pyspark
  - Stop Words : Removed commonly used words unrelated to the categories. Used StopWordsRemover API from pyspark
  - Count Vectors : Used CountVectorizer to count the frequency of each word occurring in an article (Similar to term frequency)
  - String Indexer : Converted every category to an integer label using StringIndexer from pyspark
  - IDF : Used IDF API to calculate
 the frequency of each word in a category
4. Splitting of Dataframe   
Used pyspark's random-split API to split the data to training data(80%) and test data(20%)
5. Multi Class Classification   
  - Logistic Regression : Used pyspark's LogisticRegression API to create a LR model using the training data. The labels of the test data were predicted using the trained LR Model.
###### Logistic Regression Classification pipeline:
![cap](https://user-images.githubusercontent.com/40739455/53542366-22ad8200-3aec-11e9-98a0-8eca20cc255c.JPG)
6. Accuracy   
The accuracy of the classification model was determined using the MulticlassClassificationEvaluator API from pyspark by comparing the predicted labels from the classification model and the test data's labels
7. Testing   
An unknown set of labelled data was collected and classified using the steps and the accuracy was determined

### Output:
1. Test Data Accuracy  for large datasets 
    - Logistic Regression : 71.92%
2. Unknown data Accuracy
    - Logistic Regression : 75.44%

