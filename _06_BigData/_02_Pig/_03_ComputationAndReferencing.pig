/* ComputationAndReferencing.pig
This code will load a CSV file (dataset) and then perform some compuations on the loaded data
*/

-- Lets load the large NYSE stocks dataset
nyse_stocks = LOAD '/user/PigData/NYSE_stocks.csv' using PigStorage(',') AS (exchange:chararray, stock_symbol:chararray, date:chararray, open_price:double,
                                                                                                high_price:double, low_price:double, close_price:double, volume:int,
                                                                                                  adj_price:double);

-- Now we will compute the price change
price_change = FOREACH nyse_stocks GENERATE (close_price - open_price) AS close_open, (low_price - high_price) AS high_low;

-- we will write the result of the computation into a path in HDFS as follows, in the format of CSV
-- If we do not mention the format to PigStorage, it will by default write a tab delimited file
STORE price_change INTO '/user/PigData/price_change_output' using PigStorage(',');


