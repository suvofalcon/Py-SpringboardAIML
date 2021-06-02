
# Import Spark Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.functions import min
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Create the spark session and initialize app name
spark = SparkSession.builder.appName("MinTemperature").getOrCreate()

# We define a schema which has to be a struct type
schema = StructType([StructField("stationID", StringType(), True),
                    StructField("timestamp", IntegerType(), True),
                    StructField("measure_type", StringType(), True),
                    StructField("temperature", FloatType(), True)])

# We will load the data and read as dataframe
df = spark.read.schema(schema).csv("../resources/1800.csv")

# Show the first 20 entries and the schema
df.show()
df.printSchema()

# We will filter out all, except "TMIN" entries
minTemps = df.filter(df.measure_type == "TMIN")
# Now we will select only Station ID and temperature
minTemps = minTemps.select("stationID", "temperature")

# Aggregate to find the minimum tempeature by stationID
minTemps = minTemps.groupBy("stationID").agg(min("temperature").alias("MinTemperature"))
minTemps.show()

# OR another approach
# minTemps = minTemps.groupBy("stationID").min("temperature").withColumnRenamed("min(temperature)", "MinTemperature")
# minTemps.show()

# Convert temperature to fahrenheit and sort the dataset
# withColumn creates a new column with the computed values
minTempsByStation = minTemps.withColumn("MinTempFarenh",
                                                  func.round(func.col("MinTemperature") * 0.1 * (9.0 / 5.0) + 32.0, 2))\
                                                  .select("stationID", "MinTempFarenh").sort("MinTempFarenh")

minTempsByStation.show()
# End the spark session
spark.stop()
