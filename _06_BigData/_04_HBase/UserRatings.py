"""
Ratings by Userid using HBase
"""

from starbase import Connection

# create the connection object
conn = Connection("127.0.0.1", "16040")

# We will create that schema

ratings = conn.table('ratings')

if(ratings.exists()):
    print("Dropping existing ratings table\n")
    ratings.drop()

ratings.create('rating')

print("Parsing the ml-100k ratings data...\n")
ratingFile = open("resources/ml-100k/u.data", "r")

batch = ratings.batch()

# Unique Row id is the userID
# rating column family is going to populate itself with a rating column for the movie id with a rating value
for line in ratingFile:
    (userID, movieID, rating, timestamp) = line.split()
    batch.update(userID, {'rating': {movieID: rating}})

ratingFile.close()

print("Committing ratings data to HBase via REST service\n")
batch.commit(finalize=True)

print("Get back ratings for some users...\n")
print("Ratings for user ID 1:\n")
print(ratings.fetch("1"))
print("Ratings for user ID 2:\n")
print(ratings.fetch("33"))

ratings.drop()

