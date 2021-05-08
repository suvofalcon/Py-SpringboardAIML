/*
- Load the salary file and declare its structure.
- Loop through the input data to clean up the number fields. Take out the commas from the salary and travel fields and cast to a float.
- Trim down to just Local Boards of Education
- Further trim it down to just be for the year in question
- Bucket them up by the job title.
- Loop through the titles and check how many are there under each title.
- Determine the minimum, maximum and average salaries for every title.
- Guarantee the order on the way out.
- Dump the results on the console.
- Save results back to HDFS.
 */

-- We will register the piggybank.jar first
register '/home/hduser/Cluster/pig-0.16/contrib/piggybank/java/piggybank.jar';
-- From the piggybank repository, we will use the REPLACE function
define REPLACE org.apache.pig.piggybank.evaluation.string.REPLACE();

-- We will load the CSV file. The CSV file has comma used within the fields and also as delimeters. We will use the
-- CSVExcelStorage() function.
define CSVExcelStorage org.apache.pig.piggybank.storage.CSVExcelStorage();

-- Load the file
travelFile = LOAD '/user/PigData/salaryTravelReport_sample.csv' using CSVExcelStorage() AS (EmployeeName:chararray, JobTitle:chararray, Salary:chararray, TravelAllowance:chararray, OrganizationType:chararray, Organization:chararray, FiscalYear:int);

-- We will clean up the input data by removing ',' from Salary and TravelAllowance and cast it to float
travelFile_cleaned = FOREACH travelFile GENERATE EmployeeName AS EmployeeName, JobTitle AS JobTitle, (float) REPLACE(Salary,',','') AS Salary, (float) REPLACE(TravelAllowance,',','') AS TravelAllowance, OrganizationType AS OrganizationType, Organization AS Organization, FiscalYear as FiscalYear;
-- Filter it down to just Local Boards of Education
travelFile_cleaned_LBOE = FILTER travelFile_cleaned BY OrganizationType == 'LBOE';
-- Filter it down further to just FiscalYear 2010
travelFile_cleaned_LBOE_2010 = FILTER travelFile_cleaned_LBOE BY FiscalYear == 2010;

-- Bucket them by JobTitle
travelFile_TitleBucket = GROUP travelFile_cleaned_LBOE_2010 BY JobTitle;

-- We will loop through this bucket and check the count, min, max abd average salaries. All these are aggregate function which would need the preceeding GROUP BY
travelFileResults = FOREACH travelFile_TitleBucket GENERATE group AS JobTitle, COUNT(travelFile_cleaned_LBOE_2010) AS COUNT, MIN(travelFile_cleaned_LBOE_2010.Salary) AS MinSalary, MAX(travelFile_cleaned_LBOE_2010.Salary) AS MaxSalary, AVG(travelFile_cleaned_LBOE_2010.Salary) AS AvgSalary;
-- We will order it by JobTitle
travelFileResults = ORDER travelFileResults by JobTitle;
dump travelFileResults;
--STORE travelFileResults INTO '/user/PigData/SalaryResultsOutput';