-- Views in Hive
-- Views are queried just like how we query a table
-- If a query references a view, the definition of the view is combined with the rest of the query by Hive's query planned
-- No storage allocated for this view
-- A view's name must be unique compared to all other table and view names in the same database

create view vwnyse as select exchangename, stock_symbol, transaction_date, high_price from tblnyse limit 100;
-- Check the view execution
select * from vwnyse;
-- to see the views we need to use the show tables command as there is no show views command
show tables;
-- to see the structure of the view
describe formatted vwnyse;

-- When the view and a qwury that uses it, both contains an ORDER BY clause or a LIMIT clause, the view's clauses are evaluated before using query's classes.
-- lets say we use the earlier view which has a limit of 100 and run a query to limit 200 rows - it will still return 100 rows as the view is evaluated first
select * from vwnyse limit 100;

-- Using views to reduce complexity
-- We will use the below query to be rewritten using views
select stock_symbol, case when round(close_price - open_price, 2) > 0 then "UP" when round(close_price - open_price, 2) < 0 then "DOWN" else "NOCHANGE" end as 
        ChangeStatus, count(*) from tblnyse 
        group by stock_symbol, case when round(close_price - open_price, 2) > 0 then "UP" 
        when round(close_price - open_price, 2) < 0 then "DOWN" 
        else "NOCHANGE" end;
        
-- changing a complicated query to a view
create view vwsymbolpricechange as 
    select stock_symbol, case when round(close_price - open_price, 2) > 0 then "UP" 
    when round(close_price - open_price, 2) < 0 then "DOWN" else "NOCHANGE" end as 
    ChangeStatus from tblnyse;
-- check the view
select * from vwsymbolpricechange;

-- using that view to simplify the query
select ChangeStatus, count(*) from vwsymbolpricechange group by ChangeStatus;

--- Hive Joins
---------------------------------------

-- Inner Joins - Only those records are retained that have a match in every table being joined
-- Query to find out which companies issued dividends
select a.stock_symbol, a.close_price from tblnyse a join tblnysedividends b on a.stock_symbol = b.stock_symbol;
-- can we re-written without using alias
select tblnyse.stock_symbol, tblnyse.close_price from tblnyse join tblnysedividends where tblnyse.stock_symbol = tblnysedividends.stock_symbol;
-- hive does not currently support using OR between predicates in ON clauses
-- we can also use where clauses in join
select a.stock_symbol, a.close_price from tblnyse a join tblnysedividends b on a.stock_symbol = b.stock_symbol where a.close_price >= 10;

-- only equi-joins are allowed in hive currently ( in the "on" condition only = sign can be used)
-- Hive can join a table to itself and is called self join

-- we can also join more than two tables together and a sample query is as below
select a.stock_symbol, a.close_price, b.dividends, c.economoy from tblnyse a
    join tblnysedividends b on a.exchangename = b.exchangename and a.stock_symbol = b.stock_symbol
    join tblstockexchanges c on b.exchangename = c.exchangename;
-- hive always goes from left to right while joining

-- instead of referencing the entire table during joins, we can also replace a table with a query (this tremendously increases performance)
select a.stock_symbol, a.close_price, b.dividend from tblnyse a
    join (select stock_symbol, dividend from tblnysedividends) b on a.stock_symbol = b.stock_symbol
    where b.dividend <= 10;

-- hive assumes that the last table in the query is the largest
-- it attempts to buffrer the other tables and then stream the last table through, while performing joins on individual records. Hence it
-- is advisable to put the largest table at the last while working on inner joins in hive

-- Hive Outer join
-- left outer join
-- all records from left hand table will be retained, even if there is no match
-- if the right hand table does not have a match, NULL us used for each column in the right hand table
-- if there is a where condition on the join statement, where conditions will be evaluated after the join is performed.

select a.stock_symbol, a.close_price, b.dividend from tblnyse a
    left outer join tblnysedividends b on a.stock_symbol = b.stock_symbol;
    
-- Similarly right outer join
select a.stock_symbol, a.close_price, b.dividend from tblnyse a
    right outer join tblnysedividends b on a.stock_symbol = b.stock_symbol;
-- All records from right hand table will be retained, even if there is no match
-- if right hand table does not have a match in the left hand table, NULL is used for each column in the left hand table

