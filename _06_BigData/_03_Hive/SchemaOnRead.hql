-- Hive maintains a “metastore” that imparts a structure you define on the
-- unstructured data that is stored on HDFS etc.

-- ■ LOAD DATA
-- MOVES data from a distributed filesystem into Hive
--■ LOAD DATA LOCAL
-- COPIES data from your local filesystem into Hive

-- ■ Managed vs. External tables

CREATE EXTERNAL TABLE IF NOT EXISTS ratings (
userID INT, movieID INT, rating INT, time INT)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ‘\t’
LOCATION ‘/data/ml-100k/u.data’;

CREATE TABLE ratings (
userID INT, movieID INT, rating INT, time INT)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ’\t’
STORED AS TEXTFILE;

LOAD DATA LOCAL INPATH ‘${env:HOME}/ml-100k/u.data’
OVERWRITE INTO TABLE ratings;