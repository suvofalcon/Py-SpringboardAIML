-- Ways to Query Hive tables
select exchangename, stock_symbol, dividend from dbnyse.tblnysedividends;
select name, marketcap, countries[1], cxos["CEO"], cxos["CFO"], hqaddress.street from dbsample.tblcompany;

-- Manipulate column values using Operators
select upper(name), marketcap/1000 AS MarketCapInBillions from dbsample.tblcompany;

-- cast converts one data type to another
select name, cast(marketcap as string) from dbsample.tblcompany;

-- Table Generating functions
-- They take single columns and expand them to multiple columns or rows
select explode(countries) from dbsample.tblcompany;
-- explode takes an array or map as input and output the elements of the array or map as rows
select posexplode(countries) from dbsample.tblcompany;
-- posexplode takes an array or map as input and output the elements of the array or map with its position as rows

-- Lateral view is used in conjunction with user defined table generating functions
        -- Applies the UDTF to each row of the base table and generates output
select a.name, b.country from dbsample.tblcompany a lateral view explode(a.countries) b as country;
-- Output of the lateral view which is a virtual table is joined with the input rows of the tblcompany to get the result

-- mutiple lateral views can be used as well.. for example if we want to output the name, country(each element) and the CXOs
select a.name, b.country, c.role, c.name from dbsample.tblcompany a lateral view explode(a.countries) b as country 
                lateral view explode(a.cxos) c as role, name;

-- case statement (similar to if else statement)
select stock_symbol, case when round(opencloseadj["close"] - opencloseadj["open"],2) > 0 then "UP" else "DOWN" end from dbnyse.tblnysedaily;
-- using multiple case statements
select stock_symbol, case when round(opencloseadj["close"] - opencloseadj["open"],2) > 0 then "UP" 
                            when round(opencloseadj["close"] - opencloseadj["open"],2) < 0 then "DOWN" else "NOCHANGE" end 
                            change from dbnyse.tblnysedaily;

-- group by statement to use aggregate information
-- how wmany stocks went up and down and no change using group by
select case when round(opencloseadj["close"] - opencloseadj["open"],2) > 0 then "UP"
        when round(opencloseadj["close"] - opencloseadj["open"],2) < 0 then "DOWN"
        else "NOCHANGE" end as stock_status, count(*) from dbnyse.tblnysedaily
        group by case when round(opencloseadj["close"] - opencloseadj["open"],2) > 0 then "UP"
        when round(opencloseadj["close"] - opencloseadj["open"],2) < 0 then "DOWN"
        else "NOCHANGE" end
-- The case statements has to be repeated in the group by because hive does not allow to reference column aliases in the where clause
-- and in the group by clauses.

-- Find the average stock prices by stocks and by year
select upper(stock_symbol) as stock_symbol, substr(transaction_date,length(transaction_date)-3) as transaction_year,
                                            AVG(close_price) from dbnyse.tblnyse 
                                            group by upper(stock_symbol), substr(transaction_date,length(transaction_date)-3);

-- group where average close price is greater than 30
select upper(stock_symbol) as stock_symbol, substr(transaction_date,length(transaction_date)-3) as transaction_year,
                                            AVG(close_price) from dbnyse.tblnyse 
                                            group by upper(stock_symbol), substr(transaction_date,length(transaction_date)-3)
                                            having AVG(close_price) > 30;
                                            
-- Code to demonstrate nested select
select * from (
    select * from dbnyse.tblnyse where stock_symbol = 'AEP'
    union all
    select * from dbnyse.tblnyse where stock_symbol = 'AAP') A;
    
-- We can also use subsqueries
select * from dbnyse.tblnyse d where d.stock_symbol in 
    (select stock_symbol from dbnyse.tblnysedividends where transaction_date = '2009-11-12');

