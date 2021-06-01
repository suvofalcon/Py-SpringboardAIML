
# Import the Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func

# Create the spark session
spark = SparkSession.builder.appName("FriendsByAge").getOrCreate()

# Load the data as spark dataframe
people = spark.read.option("header", "true").option("inferSchema", "true").csv("../resources/fakefriends-header.csv")

# Select only the attributes needed
people = people.select("age", "friends")

# Lets do the grouping and sorting 
# We will add a custom name column
people.groupBy("age").agg(func.round(func.avg("friends"), 2).alias("Avg_Friends")).orderBy("age").show()

# Close the spark session
spark.stop()