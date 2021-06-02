# Import the Spark libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as func

# Create the spark session
spark = SparkSession.builder.appName("WordCountDataFrame").getOrCreate()

# Read each line of the file into the dataframe
inputDF = spark.read.text("../resources/Book.txt")

# Split the line and extract words using regular expressions
words = inputDF.select(func.explode(func.split(inputDF.value, "\\W+")).alias("Word"))
words.filter(words.Word != "")

# Convert all words to lowercase
lowerCaseWords = words.select(func.lower(words.Word).alias("Word"))
lowerCaseWords.show()

# Now we will do a group by aggregate using count
wordCountSorted = lowerCaseWords.groupBy("Word").count().sort("count")

# Show the results
wordCountSorted.show(wordCountSorted.count())

spark.stop()