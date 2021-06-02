
# Spark Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructField, StructType, StringType, FloatType, IntegerType


# Create the spark session and initialize the app name
spark = SparkSession.builder.appName("TotalSpentByCustomers").getOrCreate()

# Build the custom schema
schema = StructType([StructField("CustomerID", IntegerType(), False),
                    StructField("ItemID", StringType(), True),
                    StructField("Amount", FloatType(), True)])

# Load the data and read as dataframe
df = spark.read.schema(schema=schema).csv("../resources/customer-orders.csv")

# Verify the data load and show the schema
df.show(10) # We will only show top 10 rows
df.printSchema()

# we will extract only he relevant fields
dfCustomerSpent = df.select("CustomerID", "Amount")

# Now we will perform the aggregation
dfCustomerSpent = dfCustomerSpent.groupBy("CustomerID").agg(func.round(func.sum("Amount"), 2).alias("TotalSpent"))

# We will sort the results
dfCustomerSpent = dfCustomerSpent.sort("TotalSpent")

# Show the results
dfCustomerSpent.show(dfCustomerSpent.count())

spark.stop()



