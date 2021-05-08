/* FilterDemo.pig
This code will load a CSV file and apply a filter based on certain conditions
*/

-- Lets load the file
-- PigStorage can be used to load various types of files
-- PigStorage (',') denotes that the value is delimited with ','
nyse_dividends = LOAD '/user/PigData/NYSE_dividends_A.csv' using PigStorage(',') AS (exchange:chararray, symbol:chararray, date:chararray, dividends:float);
-- We will apply a filter criteria and would have only those records in the relation where dividend is 0
nyse_filtered = FILTER nyse_dividends BY dividends == 0.00;
--dump nyse_filtered;

-- Now for all the stocks which has not given dividends, lets convert the stock symbol to lower and output the same
nyse_lower = FOREACH nyse_filtered GENERATE LOWER(symbol) as lower_stockSymbols;
dump nyse_lower;