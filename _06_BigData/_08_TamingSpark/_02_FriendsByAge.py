# Import the Spark contexr

from pyspark import SparkConf, SparkContext
import collections

# Create the Spark Context
conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf=conf)

# function to extract age and num of friends and return

def parseRecords(records):
    fields = records.split(',')
    age = int(fields[2])
    numOfFriends = int(fields[3])
    return (age, numOfFriends)

# Load the file
records = sc.textFile("../resources/fakefriends.csv")
# transform the rdd
rdd = records.map(parseRecords)

'''
rdd.mapValues(lambda x: (x, 1))
(33, 385) => (33,(385, 1))
(33, 2) => (33, (2, 1))
(55, 2) => (55, (2, 1))

reduceByKey(lambda x, y : (x[0] + y[0], x[1] + y[1]))
Adds up all values for each unique key

(33, (387, 2))
x = (385, 1)
y = (2, 1)

This will continue by each key, till all of the keys are resolved
'''
# we will find totals by age
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y : (x[0] + y[0], x[1] + y[1]))

'''
mapValues(lambda x: x[0]/x[1])
(33, (387, 2)) => (33, 193.5)

387 - Total number of friends for the age 33 (x[0])
2 - Number of times the age 33 came up (x[1])
'''
# We will now find the average
averageByAge = totalsByAge.mapValues(lambda x: round(x[0]/x[1], 0))

# Now the results are collected
# This where spark will compute the DAG and find out the most optimum way to compute the info
results = averageByAge.collect() 

# Now we will iterate the print each row
for item in results:
    print(item);





