
# Get the Spark libraries
from pyspark import SparkConf, SparkContext

# Create the Spark context and set the app name
conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")
sc = SparkContext(conf=conf)

# Defining a function to parse and extract relevant fields

def parseRecords(line):
    fields = line.split(",")
    stationID = fields[0]
    entryType = fields[2]
    # temperature convert to celcius
    temperature = round(float(fields[3]) * 0.1 * (9.0/5.0) + 32, 2)
    return(stationID, entryType, temperature)

# Load the data
data = sc.textFile("../resources/1800.csv")
# We will use the custom function to extract the records
records = data.map(parseRecords)

# Print the first few entries in the rdd
# print(records.take(10))

# From the RDD, we will only filter out those entries which have TMIN in entryType
'''
Hence it will look like
('ITE00100554', 'TMAX', 18.5) - after filter
('ITE00100554', 18.5) - after map
'''
tempsByStation = records.filter(lambda x: "TMAX" in x[1]).map(lambda x: (x[0], x[2]))
print(tempsByStation.take(5))

# Now we will reduce by key
maxTempsByStation = tempsByStation.reduceByKey(lambda x, y: max(x, y))
maxTempsByStation = maxTempsByStation.collect()

# We will print the result
for items in maxTempsByStation:
    print(f"Station ID - {items[0]}, max Temperature - {items[1]}")