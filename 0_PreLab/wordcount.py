import sys

from pyspark import SparkConf, SparkContext

from operator import add

## Constants
APP_NAME = " HelloWorld of Big Data"
##OTHER FUNCTIONS/CLASSES

def main(sc,filepath):
   textRDD = sc.textFile(filepath)
   words = textRDD.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1))
   wordcount = words.reduceByKey(add).collect()
   for wc in wordcount:
      print wc

if __name__ == "__main__":

   # Configure Spark
   conf = SparkConf().setAppName(APP_NAME)
   conf = conf.setMaster("local[*]")
   sc   = SparkContext(conf=conf)
   textfile = sys.argv[1]
   # Execute Main functionality
   main(sc,textfile)