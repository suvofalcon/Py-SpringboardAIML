
# Import Spark Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import LongType, StructType, StructField, IntegerType
import codecs

# Create the spark session and initialize the app name
spark = SparkSession.builder.appName("MostPopularMovieNames").getOrCreate()

'''
This function will load movie names from u.item file and return a dictionary
'''

def loadMovieNames():
    movieNames = {}

    # We will load from the u.item file
    with codecs.open("../resources/ml-100k/u.item", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames

'''
Create a user defined functions to look up the movie names from our broadcasted dictionary
'''
def lookUpMovieName(movieID):
    return nameDict.value[movieID]

'''
We will now use broadcast method. 
This broadcast objects to the executors, such that they are always there when needed
Very beneficial for large data
'''
nameDict = spark.sparkContext.broadcast(loadMovieNames())

# Initialize the Schema - This schema will be used to load the u.data
schema = StructType([StructField("UserID", IntegerType(), False),
                    StructField("MovieID", IntegerType(), True),
                    StructField("Rating", IntegerType(), True),
                    StructField("Timestamp", LongType(), True)])

# Load the data and create a dataframe and assign the schema
movieData = spark.read.option("sep", "\t").schema(schema=schema).csv("../resources/ml-100k/u.data")

# Now movie counts by ID
movieCounts = movieData.groupBy("MovieID").count()

# We will initialize a user defined function
lookUpNameUDF = func.udf(lookUpMovieName)

# We will now add a new column in our dataframe using this udf
movieCounts = movieCounts.withColumn("movieTitle", lookUpNameUDF(func.col("MovieID")))

# We will sort the results
movieCounts = movieCounts.orderBy(func.desc("count"))

# we will show the top 10
movieCounts.show(10)

# Close the spark session
spark.stop()


