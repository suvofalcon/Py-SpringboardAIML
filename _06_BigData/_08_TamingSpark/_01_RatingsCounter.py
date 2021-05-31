
# Import the Spark contexr

from pyspark import SparkConf, SparkContext
import collections

'''
Spark Context, cannnot be created without SparkConf
Spark Conf, tells where we want to run, = Local/ Cluster 
'''

# Set the master node as the local machine, running on the local box and not on cluster
# There are extensions on local, which tells us to split it on multiple machines
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf=conf)

# We will load the dataset now (we will load from local file system) and this will return a RDD
records = sc.textFile("../resources/ml-100k/u.data")

# Extract the ratings from each of the records .. (ratings is third column)
# We will achieve this using mapper function
ratings = records.map(lambda x: x.split()[2])

# Now we will call an action on the RDD
results = ratings.countByValue()

# Now we will sort the results dictionary
# This gets sorted by default on the keys
sortedResults = collections.OrderedDict(sorted(results.items()))

# we will print the results - iterating every key, value pair in the dictionary
for key, value in sortedResults.items():
    print("%s %i" %(key, value))





