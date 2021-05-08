/*
Various Codes to demonstrate Parameter Substitution
Pig Provides options to pass values as parameters. It does so by using String replacement functionality
Parameters can be passed on command line or from a file

This file has to be saved as a script and run from command line as
pig -p date=17-12-2011 ParamSubs.pig

if Params are getting substituted from a file then

 */

-- PigLatin script to find stock prices by date. Date will be passed through command line
nyse_stocks = LOAD '/user/PigData/NYSE_stocks.csv' using PigStorage(',') AS (exchange:chararray, stock_symbol:chararray, date:chararray, open_price:double,
                                                                                                high_price:double, low_price:double, close_price:double, volume:int,
                                                                                                  adj_price:double);
stock_prices = FILTER nyse_stocks BY date == '$date';
dump stock_prices;

-- This time we will pass the parameters from a file
-- Lets say, we want to find out all stock prices by date and those that are greater than a certain threshold value
--stock_prices_file = FILTER nyse_stocks BY date == '$date' AND close_price > $threshold;
--dump stock_prices_file;

-- We can pass paramters from the command line and from the file together at the same time. In that case params passed from command line
-- will take precedence