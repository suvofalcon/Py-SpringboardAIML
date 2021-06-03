
# Load the spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import LongType, StringType, StructType, StructField, IntegerType

# Initialize the session and set the application name
spark = SparkSession.builder.appName("MostObscureHeros").getOrCreate()

# Define the custom schema to read the name file
schema = StructType([StructField("HeroID", IntegerType(), True),
                    StructField("HeroName", StringType(), True)])

# Read the name file using the custom schema
names = spark.read.schema(schema=schema).option("sep", " ").csv("../resources/Marvel-Names")

# Load the entries data (without any schema)
lines = spark.read.text("../resources/Marvel-Graph")

'''
Split off HeroID from the beginning of  the line
Count how many space-separated numbers are in the line
- Subtract 1 to get hold of total number of connections
Group by HeroIDs to add up connections split into multiple lines
Sort by total connections
'''

# we need to trim each line of whitespaces as that could throw off the count
connections = lines.withColumn("HeroID", func.split(func.trim(func.col("value")), " ")[0])\
                                .withColumn("Connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1)\
                                    .groupBy("HeroID").agg(func.sum("Connections").alias("Connections"))

# Sort by ascending number of connections
connections = connections.orderBy(func.asc("Connections"))
# Show the least 10 - starting from one with the lowest
connections.show(10)

# We will extract the first entry - MovieID
minConnectionCount = connections.agg(func.min("Connections")).first()[0]

minConnections = connections.filter(func.col("Connections") == minConnectionCount)
minConnectionsWithNames = minConnections.join(names, "HeroID")

print(f" The following characters have only {str(minConnectionCount)} connections")
minConnectionsWithNames.select("HeroName").show()

spark.stop()