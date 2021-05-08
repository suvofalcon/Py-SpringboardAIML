/*
Design a program that will accept start time and end time from the user. The program should tell the list of all the
users and the number of times each user logged in during the given time period.
 */

-- We will load the access logs
accesslog_1 = LOAD '/user/PigData/accesslog1' using PigStorage(' ') AS (User:chararray, duration:int);
accesslog_2 = LOAD '/user/PigData/accesslog2' using PigStorage(' ') AS (User:chararray, duration:int);

-- Join the two datasets
accesslog = UNION accesslog_1, accesslog_2;

-- We will filter this dataset based on, only the users who has logged in between the start and end time specified
-- by the user from the command line
accesslog_filtered = FILTER accesslog BY duration >= $start_time AND duration <= $end_time;
--dump accesslog_filtered;

-- We will now group this dataset by users and use COUNT to see how many times the user has logged in between that
-- period
accesslog_filtered_grouped = GROUP accesslog_filtered BY User;
accesslog_results = FOREACH accesslog_filtered_grouped GENERATE group AS User, COUNT(accesslog_filtered);
dump accesslog_results;
