/* FilterDemo.pig
This code will load demonstrate all relational operators in Pig in an exhaustive manner
*/

-- Lets load the complete NYSE stocks file
nyse_stocks = LOAD '/user/PigData/NYSE_stocks.csv' using PigStorage(',') AS (exchange:chararray, stock_symbol:chararray, date:chararray, open_price:double,
                                                                                                high_price:double, low_price:double, close_price:double, volume:int,
                                                                                                  adj_price:double);

-- If we need to get only Symbol, open and close prices
nyse_subset = FOREACH nyse_stocks GENERATE stock_symbol, open_price, close_price;
-- dump nyse_subset;

-- to add a new output, whether the share price has been Up, Down or 'No Change'
nyse_change_flag = FOREACH nyse_subset GENERATE stock_symbol, ((close_price == open_price)? 'No Change' :((close_price > open_price)? 'UP' : 'DOWN'));
-- dump nyse_change_flag

-- Filter allows which records to be retained
nyse_filtered = FILTER nyse_stocks BY close_price > 150; -- This is stocks where closing price is greater than 150
-- dump nyse_filtered
-- stocks which are closing 2 percent higher
nyse_two_higher = FILTER nyse_stocks BY (close_price - open_price)/ 100 > 2;
-- dump nyse_two_higher

-- to get all stocks which symbol starts with AB
starts_with_AB = FILTER nyse_stocks BY stock_symbol MATCHES 'AB*';
-- dump starts_with_AB

-- GROUP operator, Collects records together using a key. However unlike SQL, GROUP does not need a mandatory aggregate function
-- We will use GROUP to find count of records in each group
grouped_data = GROUP nyse_stocks BY stock_symbol;
count_grouped_data = FOREACH grouped_data GENERATE group, COUNT(nyse_stocks);
-- dump count_grouped_data;
-- We can also group on multiple keys
nyse_dividends_A = LOAD '/user/PigData/NYSE_dividends_A.csv' using PigStorage(',') AS (exchange:chararray, stock_symbol:chararray,
                                                                                    date:chararray, dividends:double);
-- We can group on multiple keys
grouped_data_multiple = GROUP nyse_dividends_A BY (exchange, stock_symbol);
avg_dividend = FOREACH grouped_data_multiple GENERATE group, AVG(nyse_dividends_A.dividends) AS avg_dividend;
-- dump avg_dividend;

-- ORDER sorts the data to produce total order of the data
orderDividends = ORDER nyse_dividends_A by dividends DESC;
-- dump orderDividends

-- DISTINCT displays the unique only
-- In Pig, DISTINCT can only be applied to all fields in a relation unlike SQL
distinctSymbol = DISTINCT nyse_dividends_A;
-- dump distinctSymbol;
-- So to find out the distinct stock symbol, we follow the steps as below
distinctSymbol = LOAD '/user/PigData/NYSE_dividends_A.csv' using PigStorage(',') AS (exchange:chararray, stock_symbol:chararray);
uniqueSymbol = DISTINCT distinctSymbol;
-- dump uniqueSymbol;

-- LIMIT operator Limits the number of records in a relation
input_dividends = LIMIT nyse_dividends_A 25;
-- dump input_dividends;

-- JOIN selects record from one dataset and puts it together with another data set based on a key
-- This is also a INNER join and is default in Pig
-- JOIN can happen on multiple keys
-- Multiple datasets can be joined based on keys as long as the keys represent the same field
customers = LOAD '/user/PigData/customers.csv' using PigStorage(',') AS (customerid:int, name:chararray);
orders = LOAD '/user/PigData/orders.csv' using PigStorage(',') AS (orderid:int, orderno:int, customerid:int);
joined_dataset = JOIN customers BY customerid, orders BY customerid;
-- dump joined_dataset;

-- If we need to find the list of customers who have placed more than one order
-- We already have a joined_dataset and we will do a group by there
-- This group will essentially create a key-value pair, where key will be customerid of type int and value will be entire joined_dataset
-- On this key-value pair we will run COUNT for every key and filter out those which is of more than 1
group_joined_dataset = GROUP joined_dataset BY customers::customerid;
grouped_joined_dataset_filtered = FILTER group_joined_dataset BY COUNT(joined_dataset) > 1;
-- dump grouped_joined_dataset_filtered;
-- In addition to all above Pig also supports, Left Outer, Right Outer and Full Outer joins. Please refer notes for the same

-- COGROUP - combines two or more datasets by a column and join based on the same column
-- COGROUP on one dataset is same as that of GROUP
-- COGROUP on multiple datasets results in a record with a key and one bag per dataset
cogrouped_dataset = COGROUP orders BY customerid, customers BY customerid;
-- dump cogrouped_dataset;

-- UNION is used to concatenate two datasets together and not joined
union_customers_orders = UNION customers, orders;
-- More on the class notes

-- RANK each tuple in a relation
-- RANK is very usesful in business scenarios when we need to know, our top N customers
sales_data = LOAD '/user/PigData/rank' AS (name:chararray, sales_numbers:int);
-- If we want to rank the sales data by sales numbers in descending order and ensure that there are no gaps in ranks
ranked_sales_data = RANK sales_data BY sales_numbers DESC DENSE;
-- to know the top 3 customers
top3_customers = FILTER ranked_sales_data BY rank_sales_data <= 3;
-- dump top3_customers;
