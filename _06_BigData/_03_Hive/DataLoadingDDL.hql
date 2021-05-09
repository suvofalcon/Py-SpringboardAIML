-- Create table syntax in Hive
CREATE TABLE IF NOT EXISTS dbhive.tblCompany (
	name STRING,
	marketcap FLOAT COMMENT 'Market Capitalization of the company',
	countries ARRAY<STRING>,
	cxos MAP<STRING,STRING> COMMENT 'Chief Executive Officers',
	hqaddress STRUCT<street:STRING, city:STRING, state:STRING, zip:INT> COMMENT 'Address of 	the company headquarters')
	
	COMMENT 'Table to represent a company'	

	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '#'
	COLLECTION ITEMS TERMINATED BY '*'
	MAP KEYS TERMINATED BY '$'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE;

-- How to create a table using schema of another table (only structure and not data)
CREATE TABLE tblCompanyCopy LIKE tblCompany;

-- To see the structure of a table
DESCRIBE tblCompany;
DESCRIBE EXTENDED tblcompany;
DESCRIBE FORMATTED tblCompany; -- More detailed description

-- create a table like another table and load data at the same time
CREATE TABLE tblcompany2 AS SELECT * FROM tblcompany;

-- create a table as a subset from another table, with data
CREATE TABLE tblcompanysubset AS SELECT name, marketcap, countries FROM tblcompany;

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

-- Loading data to a hive table from a file in the local file system
LOAD DATA LOCAL INPATH '/home/hduser/Documents/CodeMagic/bd-hivequery/companydata.txt' OVERWRITE INTO TABLE tblCompany;

-- Loading data to a hive table from a file in the HDFS file system
LOAD DATA INPATH '/user/hduser/hive/companydata' OVERWRITE INTO TABLE tblCompany;

-- Loading data from one hive table to another hive table
INSERT OVERWRITE TABLE tblcompanycopy SELECT * FROM tblcompany;

-- Hive creates a directory for every table being created and the data is stored inside those directories as files. This directory for the table is created
-- inside the database directory (a directory is also created in HDFS for every database in hive).
-- The data inside the tables can also be put by using simple hdfs file operations

-- Hence Hive can only check and enforce constraints during read and not during insertion, because Hive does not control the storage of data in Hadoop, it is
-- controlled by HDFS

-- Load data from a single tables to multiple tables using a single select statement
FROM tblcompany
INSERT OVERWRITE TABLE tblcompany2 SELECT * WHERE name='Microsoft'
INSERT OVERWRITE TABLE tblcompany3 SELECT * WHERE name='Oracle';

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

-- Exporting data from a Hive Table
INSERT OVERWRITE LOCAL DIRECTORY '/home/hduser/Documents/test.txt' SELECT name, marketcap FROM tblcompany;

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
-- Creating a managed table

CREATE TABLE tblnysedaily(
	exchangename	STRING,
	symbol		STRING,
	transactiondate	STRING,
	opencloseadj	MAP<STRING,FLOAT>,
	highlow		STRUCT<high:FLOAT,low:FLOAT>,
	volume		INT
	)
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	COLLECTION ITEMS TERMINATED BY '#'
	MAP KEYS TERMINATED BY '*'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE;

-- Creating an External table

CREATE EXTERNAL TABLE tblnysedividends_A (
	exchangename	STRING,
	symbol		STRING,
	transactiondate	STRING,
	dividend	FLOAT
	)
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY ','
	LINES TERMINATED BY '\n'
	LOCATION '/user/datafiles/NYSE_dividends_A';

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------	

-- Creating Partitioned Tables

CREATE TABLE tblnysedailypartitioned (
	exchangename	STRING,
	symbol		STRING,
	transactiondate	STRING,
	opencloseadj	MAP<STRING,FLOAT>,
	highlow		STRUCT<high:FLOAT,low:FLOAT>,
	volumne		int
	)
	PARTITIONED BY (Year STRING)
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	COLLECTION ITEMS TERMINATED BY '#'
	MAP KEYS TERMINATED BY '*'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE;

-- to see partitions
SHOW PARTITIONS tblnysedailypartitioned;

-- FOR MANAGED TABLES, WE CREATE PARTITIONS BY LOADING DATA INTO THEM
LOAD DATA INPATH '/user/datafiles/nyse_daily_2010' OVERWRITE INTO TABLE tblnysedailypartitioned PARTITION (Year=2010);
LOAD DATA INPATH '/user/datafiles/nyse_daily_2011' OVERWRITE INTO TABLE tblnysedailypartitioned PARTITION (Year=2011);
-- The respective data should be present in the directories nyse_daily_2010 and nyse_daily_2011

-- ANOTHER WAY OF CREATING PARTITIONS AND LOADING DATA IS USING INSERT SELECT COMMAND
INSERT OVERWRITE TABLE tblnysedailypartitioned PARTITION (Year=2008) SELECT * FROM tblnysedaily2008;
-- OR IF WE WANT TO INSERT FROM A FILE
LOAD DATA INPATH '/user/datafiles/nyse_daily_2008' INTO TABLE tblnysedailypartitioned PARTITION(Year=2008);

-- To create dynamic partitions and load data from a table
INSERT OVERWRITE TABLE tblnysedailypartitionedcopy PARTITION (Year) SELECT exchangename, symbol, transactiondate, opencloseadj, highlow, volume, YearValue FROM tblnysedailystaging;
-- Please remember the last column of the source table will be considered as the partitioned column, also we need to set the dynamic partition to be -- true, which by default is set to false to prevent a bad query create lot of partitions
SET hive.exec.dynamic.partition.mode=nonstrict;

-- Creating a partitioned external table (Location need not be specified)
CREATE EXTERNAL TABLE tblnysedailyexternal (
	exchangename	STRING,
	symbol		STRING,
	transactiondate	STRING,
	opencloseadj	MAP<STRING,FLOAT>,
	highlow		STRUCT<high:FLOAT,low:FLOAT>,
	volume		INT
	)
	PARTITIONED BY (Year STRING)
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	COLLECTION ITEMS TERMINATED BY '#'
	MAP KEYS TERMINATED BY '*'
	LINES TERMINATED BY '\n';

-- Now to add partitions and load the data
ALTER TABLE tblnysedailyexternal ADD PARTITION(Year=2008) LOCATION '/user/datafiles/nyse_daily_2008';
ALTER TABLE tblnysedailyexternal ADD PARTITION(Year=2010) LOCATION '/user/datafiles/nyse_daily_2010';
ALTER TABLE tblnysedailyexternal ADD PARTITION(Year=2011) LOCATION '/user/datafiles/nyse_daily_2011';

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
 

