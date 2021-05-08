-- Registering a UDF
-- Assumed we have registered the piggybank.jar
-- From the Grunt shell register '/home/hduser/pig-0.12.0/contrib/piggybank/java/piggybank.jar'

-- http://svn.apache.org/viewvc/pig/trunk/contrib/piggybank/java/src/main/java/org/apache/pig/piggybank/

register '/home/hduser/pig-0.16.0/contrib/piggybank/java/piggybank.jar';
-- define an alias for the user defined function
define reverse org.apache.pig.piggybank.evaluation.string.Reverse();
-- LOAD the input file from the HDFS
input_file = LOAD '/user/hduser/PigScripts/data/reverse.txt' AS (value:chararray);
reversed = FOREACH input_file GENERATE reverse($0);
dump reversed;


LOAD '/user/hduser/PigData/salaryTravelReport_sample.csv' AS (EmployeeName:chararray, JobTitle:chararray,
                                                                Salary:chararray, TravelAllowance:chararray, OrganisationType:chararray,
                                                                Organisation:chararray, FiscalYear:int);