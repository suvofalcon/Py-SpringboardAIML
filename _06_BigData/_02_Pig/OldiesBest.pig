-- The Objective is to fund the oldest 5 star movies

-- Load the ratings from ratings dataset
ratings = LOAD '/user/maria_dev/ml-100k/u.data' AS (userID:int, movieID:int, rating:int, ratingTime:int);
-- Check the load
-- DUMP ratings;

-- Load the movie details from items dataset
metadata = LOAD '/user/maria_dev/ml-100k/u.item' USING PigStorage('|') 
AS (movieID:int, movieTitle:chararray, releaseDate:chararray, videoRelease:chararray, imdbLink:chararray);
-- Check the load   
-- DUMP metadata;

-- From metadata load, extract MovieID, MovieTitle and releaseDate and convert the same to timestamp
nameLookup = FOREACH metadata GENERATE movieID, movieTitle, 
		ToUnixTime(ToDate(releaseDate, 'dd-MMM-yyyy')) AS releaseTime;
-- Check the name LookUp
-- DUMP nameLookup;

-- By MovieID, we will group the ratings
ratingsByMovie = GROUP ratings by movieID;

-- Now we will calculate the average ratings by movieID
avgRatings = FOREACH ratingsByMovie GENERATE group as movieID, AVG(ratings.rating) as avgRating;
-- DUMP avgRatings;

-- Extract out the five star ratings only (avgRatings greater than 4)
fiveStarMovies = FILTER avgRatings by avgRating > 4.0;
-- DUMP fiveStarMovies;

-- Join with items to get movie details
fiveStarWithData = JOIN fiveStarMovies By movieID, nameLookup By movieID;
-- DESCRIBE fiveStarWithData;

-- ORDER by ReleaseTime
oldestFiveStarMovies = ORDER fiveStarWithData By releaseTime;
DUMP oldestFiveStarMovies;