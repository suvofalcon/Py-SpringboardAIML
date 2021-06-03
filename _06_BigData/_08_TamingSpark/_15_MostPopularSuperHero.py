
# Load the spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import LongType, StringType, StructType, StructField, IntegerType

'''
Split off HeroID from the beginning of  the line
Count how many space-separated numbers are in the line
- Subtract 1 to get hold of total number of connections
Group by HeroIDs to add up connections split into multiple lines
Sort by total connections
Filter name look up dataset by the most popular HeroID to look up the name
'''

# Create the spark session and initialize the app name
spark = SparkSession.builder.appName("MostPopularSuperHero").getOrCreate()

# Define the schema
schema = StructType([StructField("HeroID", IntegerType(), True),
                    StructField("HeroName", StringType(), True)])

# Load the names data and associate the above schema
names = spark.read.schema(schema=schema).option("sep"," ").csv("../resources/Marvel-Names")

# Load the entries data (without any schema)
lines = spark.read.text("../resources/Marvel-Graph")

# we need to trim each line of whitespaces as that could throw off the count
connections = lines.withColumn("HeroID", func.split(func.trim(func.col("value")), " ")[0])\
                                .withColumn("Connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1)\
                                    .groupBy("HeroID").agg(func.sum("Connections").alias("Connections"))

# Sort by descending number of connections
connections = connections.orderBy(func.desc("Connections"))
# Show the top 10
connections.show(10)

# To find the most popular
mostPopular = connections.sort(func.col("Connections").desc()).first()
# We will look up the names and add
mostPopularName = names.filter(func.col("HeroID") == mostPopular[0]).select("HeroName").first()

# Print the name , ID and connections
print(f"The most Popular Hero Name is - {mostPopularName[0]}, with {mostPopular[1]} connections")

# Close the spark session
spark.stop()