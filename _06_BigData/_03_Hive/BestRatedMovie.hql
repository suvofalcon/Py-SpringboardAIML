-- Find the best rated movie (Movies which are watched most and rated most number of times)
-- Only consider movies with more than 10 ratings

CREATE VIEW IF NOT EXISTS avgRatings AS
SELECT movieID, AVG(rating) as avgRating, COUNT(movieID) as ratingCount
FROM ratings
GROUP BY movieID
ORDER BY avgRating DESC;

-- Now we will join with names table to get the movie names
SELECT n.title, t.avgRating, t.ratingCount
FROM avgRatings t JOIN names n ON t.movieID = n.MovieID
WHERE t.ratingCount > 10;