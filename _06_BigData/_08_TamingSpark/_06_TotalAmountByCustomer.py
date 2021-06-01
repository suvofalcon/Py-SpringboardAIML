# Get the Spark libraries

from pyspark import SparkConf, SparkContext, resultiterable

# Create the Spark Context , set Application NAme
conf = SparkConf().setMaster("local").setAppName("CountWordOccurrences")
sc = SparkContext(conf=conf)

def parseRecords(line):
    fields = line.split(",")
    customerID = fields[0]
    amountSpent = float(fields[2])
    return(customerID, amountSpent)

# Load the data
file = sc.textFile("../resources/customer-orders.csv")
# We will use the custom function to extract the records
records = file.map(parseRecords)

# we will find total spend by customerID
totalAmountByCustomer = records.reduceByKey(lambda x, y: round((x + y), 2))

# Now we will sort by total amount spent
totalAmountByCustomer = totalAmountByCustomer.map(lambda x: (x[1], x[0])).sortByKey()
results = totalAmountByCustomer.collect()

# Print the results
for result in results:
    print(result)

