/*
Siesmic Analytics
Read the Siesmic Data and for every region where sensors were deployed, find out the highest magnitude of the
earthquake recorded
*/

-- We will register the piggybank.jar first
register '/home/hduser/Cluster/pig-0.16/contrib/piggybank/java/piggybank.jar';

-- We will now load the CSV file. The CSV file has comma within fields and also as delimeters, we will
-- use the CSVExcelStorage() function and hence need to register piggybank.jar as above

define CSVExcelStorage org.apache.pig.piggybank.storage.CSVExcelStorage(',','NO_MULTILINE','NOCHANGE','SKIP_INPUT_HEADER');

-- Load the Siesmic Data
seismic_data = LOAD '/user/PigData/sample.csv' USING CSVExcelStorage() AS (SRC:chararray, Eqid:chararray, Version:int, Date_time:chararray, Lat:double, Lon:double, Magnitude:float, Depth:float, NST:int, Region:chararray);
--dump seismic_data;

-- We will group the data by region
seismic_data_region = GROUP seismic_data BY Region;

-- For each of the region we will find the highest magnitude of earthquake recorded
seismic_data_region_magnitude = FOREACH seismic_data_region GENERATE group AS Region, MAX(seismic_data.Magnitude) AS MaxMagnitude;
dump seismic_data_region_magnitude;


