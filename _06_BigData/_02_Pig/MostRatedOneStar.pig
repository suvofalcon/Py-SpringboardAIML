-- load the ratings data
ratings = LOAD '/datasets/ml-100k/u.data' AS (userID:int, movieID:int, rating:int, ratingTime:int);

-- Load the items(movies data)
metadata = LOAD '/datasets/ml-100k/u.item' USING PigStorage('|') 
AS (movieID:int, movieTitle:chararray, releaseDate:chararray, videoRelease:chararray, imdbLink:chararray);

-- We will lookup the movetitle against movieID
nameLookup = FOREACH metadata GENERATE movieID, movieTitle;

-- Group ratings by movieID
groupedRatings = GROUP ratings by movieID;

-- We will calculate average rating for every movieID and number of times it has been rated
averageRatings = FOREACH groupedRatings GENERATE group as movieID, AVG(ratings.rating) as avgRating,
	COUNT(ratings.rating) AS numRatings;

-- Check all the movies with bad ratings(average rating less than 2.0)
badMovies = FILTER averageRatings By avgRating < 2.0;

-- Join it with the details
namedBadMovies = JOIN badMovies by movieID, nameLookup By movieID;

-- Get the final results
finalResults = FOREACH namedBadMovies GENERATE movieTitle AS movieName, avgRating AS avgRating, 
	numRatings As numRatings;

-- Sort the final results by num of times the movie has been rated
finalResultsSorted = ORDER finalResults By numRatings DESC;

-- See the final output
DUMP finalResultsSorted;