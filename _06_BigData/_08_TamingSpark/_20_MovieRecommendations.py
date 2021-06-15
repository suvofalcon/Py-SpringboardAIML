
# Import Libraries
from pyspark.ml import recommendation
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS
import sys
import codecs

# Function to load the dataset
def loadMovieNames():
    movieNames = {}
    with codecs.open("../resources/ml-100k/u.item", "r", encoding="ISO-8859-1", errors="ignore") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# Initialize the session and set Application name
spark = SparkSession.builder.appName("ALSExample").getOrCreate()

# Define schema to load userratings data
moviesSchema = StructType([StructField("userID", IntegerType(), True),
                        StructField("movieID", IntegerType(), True),
                        StructField("rating", IntegerType(), True),
                        StructField("timestamp", LongType(), True)])

# Load both the datasets
# Rating Dataset
ratings = spark.read.option("sep", "\t").schema(moviesSchema).csv("../resources/ml-100k/u.data")
# Movies Dataset
names = loadMovieNames()

# Check the data loads
# ratings.show(5)
# names = names.items()
# print(list(names)[:5])

# Training Recommendation Model
print("Training Recommendation Model ....")

als = ALS().setMaxIter(10).setRegParam(0.01).setUserCol("userID").setItemCol("movieID").setRatingCol("rating")
model = als.fit(ratings)

# Manually Construct a dataframe of the userID we want recommendations for
userID = int(sys.argv[1])
userSchema = StructType([StructField("userID", IntegerType(), True)])
user = spark.createDataFrame([[userID, ]], userSchema)

# Now we will use the model to recommend movies for the user
recommendations = model.recommendForUserSubset(user, 10).collect()

print("Top 10 recommendations for user ID " + str(userID))

for userRecs in recommendations:
    myRecs = userRecs[1] # userRecs is (userID, [Row(movieId, rating), Row(movieID, rating)...])
    for rec in myRecs:
        movie = rec[0]
        rating = rec[1]
        movieName = names[movie]
        print(movieName + str(rating))




