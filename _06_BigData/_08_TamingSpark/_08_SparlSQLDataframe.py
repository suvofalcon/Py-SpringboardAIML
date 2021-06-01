
# Import the Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Create the spark session
spark = SparkSession.builder.appName("SparkSQLDataframe").getOrCreate()

# Now we will read the CSV and directly read as a dataframe and not intermediate RDD

people = spark.read.option("header", "true").option("inferSchema", "true").csv("../resources/fakefriends-header.csv")

print("Here is our inferred schema:")
people.printSchema()

print("Let's display the name column:")
people.select("name").show()

print("Filter out anyone over 21")
people.filter(people.age < 21).show()

print("Group by age")
people.groupBy("age").count().show()

print("Make Everyone 10 years older")
people.select(people.name, people.age + 10).show()

spark.stop()