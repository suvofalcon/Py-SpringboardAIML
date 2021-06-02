
# Import Spark Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import LongType, StructType, StructField, IntegerType

# Create the spark session and initialize the app name
spark = SparkSession.builder.appName("MostPopularMovie").getOrCreate()

# Initialize the Schema
schema = StructType([StructField("UserID", IntegerType(), False),
                    StructField("MovieID", IntegerType(), True),
                    StructField("Rating", IntegerType(), True),
                    StructField("Timestamp", LongType(), True)])

# Load the data and create a dataframe and assign the schema
movieData = spark.read.option("sep", "\t").schema(schema=schema).csv("../resources/ml-100k/u.data")

# Verify the data load and show the schema
movieData.show(10)
movieData.printSchema()

'''
Most popular movie will be decided by the number of times, each of the movie is rated
'''
# We aggregate the count based on movie ID and sort by descending order
movieDataByCount = movieData.groupBy("MovieID").count().orderBy(func.desc("count"))
# Show the top 10
movieDataByCount.show(10)

# Close the spark session
spark.stop()
