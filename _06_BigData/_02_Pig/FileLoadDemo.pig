/* FileLoadDemo.pig
This code demonstrates how to load a file from HDFS. Code assumes that file already exists in HDFS
*/

-- PigStorage can be used to load various types of files
-- PigStorage (',') denotes that the value is delimited with ','
nyse_dividends =  LOAD '/user/PigData/NYSE_dividends_A.csv' using PigStorage(',') AS (exchange:chararray, symbol:chararray, date:chararray, dividends:chararray);
dump nyse_dividends; --  dump the output

