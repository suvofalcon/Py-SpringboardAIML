-- Find the most popular movies (Movies which are watched most and rated most number of times)

CREATE VIEW IF NOT EXISTS topMovieIDs	AS
SELECT movieID, COUNT(movieID) as ratingCount
FROM ratings
GROUP BY movieID
ORDER BY ratingCount DESC;

-- Now we will join this view with the names table to get the movietitles
SELECT n.title, ratingCount
FROM topMovieIDs t JOIN names n ON t.movieID = n.movieID;