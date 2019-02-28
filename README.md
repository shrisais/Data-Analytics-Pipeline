# data_analytics_pipeline
###Environment Details:

Following are the environment chosen for this Lab:

Apache Spark run on Virtual Machine 
Scripts were written in Python
Steps to run the program:
Note: Our file directory for the python script is /spark/lab3/src/script.py and for the data is /spark/lab3/data

From the terminal, traverse to the folder spark/
Run the script by spark-submit /lab3/src/script.py
You can see the accuracy returned by the Logistic Regression and Naive Bayes on the console
Documentation
Collect data
Data was collected using NYTimesArticles-API (as used in lab2) for the following categories:
Business : Stock, Economy, Finance
Sports : NBA, BFL, Golf
Politics : President, Trump, Election
Entertainment : Met Gala, TMZ, SNL
Dataframe
Read all articles from each of the category and appended it into a spark-dataframe along with a column specifying the category name.
Feature Engineering
Tokenizer : Tokenize each article into words using space delimiter. Used RegexTokenizer API from pyspark
Stop Words : Removed commonly used words unrelated to the categories. Used StopWordsRemover API from pyspark
Count Vectors : Used CountVectorizer to count the frequency of each word occurring in an article (Similar to term frequency)
String Indexer : Converted every category to an integer label using StringIndexer from pyspark
IDF : Used IDF API to calculate the frequency of each word in a category
Splitting of Dataframe
Used pyspark's random-split API to split the data to training data(80%) and test data(20%)
Multi Class Classification
Logistic Regression : Used pyspark's LogisticRegression API to create a LR model using the training data. The labels of the test data were predicted using the trained LR Model.
Logistic Regression Classification pipeline:
alt text


