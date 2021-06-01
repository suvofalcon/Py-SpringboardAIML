
# Spark imports
from pyspark import SparkConf, SparkContext

# Create the spark context and set the app name
conf = SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf = conf)

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
# Use the custom function to extract the records
records = data.map(parseRecords)

# Print the first few entries in the rdd
# print(records.take(10))

# From the RDD, we will only filter out those entries which have TMIN in entryType
'''
Hence it will look like
('ITE00100554', 'TMIN', 5.36)
'''
minTemps = records.filter(lambda x: "TMIN" in x[1])

# We will extract the key value-pair
'''
This will return 
('ITE00100554', 5.36)
'''
stationTemps = minTemps.map(lambda x : (x[0], x[2]))

# Now we will reduce by key to get the minimum
minTemps = stationTemps.reduceByKey(lambda x, y: min(x, y))
results = minTemps.collect()

# We will now print the result
for result in results:
    print(f"Station ID - {result[0]} and minTemp - {result[1]}")





