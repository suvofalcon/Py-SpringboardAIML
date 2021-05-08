-- Codeto demonstrate built in and user defined functions in pig

-- Average
input_dividends = LOAD '/user/PigData/NYSE_dividends_A.csv' using PigStorage(',') AS (exchange:chararray, symbol:chararray, date:chararray, dividends:float);
grouped_data = GROUP input_dividends BY symbol;
-- AVG function needs a preceeding GROUP BY Statement
avg_dividends = FOREACH grouped_data GENERATE group, AVG(input_dividends.dividends);
--dump avg_dividends;

-- COUNT (computes number of elements in a bag) -- this will not count the NULL values
count_data = FOREACH grouped_data GENERATE group, COUNT(input_dividends);
--dump count_data;

-- COUNT_STAR - Similar to count function, however this also counts the NULL values and is exactly same as before
-- Both COUNT and COUNT_STAR function needs a preceeding GROUP BY statement

-- Exactly similar to the AVG and COUNT functions are the MAX, MIN, SUM.

-- CONCATENATE two fields or expressions
-- Data types of the fields or expressions should be exactly the same
concat_fields = FOREACH input_dividends GENERATE CONCAT(exchange,symbol);
--dump concat_fields;
-- We can also concatenate multiple fields
concat_multiple_fields = FOREACH input_dividends GENERATE CONCAT(CONCAT(exchange,'-'), symbol);
dump concat_multiple_fields;
