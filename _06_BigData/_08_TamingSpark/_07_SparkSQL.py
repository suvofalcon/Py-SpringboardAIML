
# Import the Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Create the spark session
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# define a function for extraction
# This may not always be the way to read a dataset using spark dataframe, but this shows how RDDs and Spark dataframes works well with each other
def mapper(record):
    fields = record.split(",")
    return Row(ID = int(fields[0]), name = str(fields[1]),
    age = int(fields[2]), numOfFriends = int(fields[3]))


# Load the data
friendsData = spark.sparkContext.textFile("../resources/fakefriends.csv")
friendsData = friendsData.map(mapper)

# Infer the schema and register the data frame as table
schemaPeople = spark.createDataFrame(friendsData).cache()
schemaPeople.createOrReplaceTempView("people")

# If we want to see the dataframe
# schemaPeople.show()

# SQL can run over dataframe that have been registered as table
teenagers = spark.sql("SELECT * FROM people where age >= 13 and age <=18")
teenagers.show()

# We can also use functions instead of SQL Queries
schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()
