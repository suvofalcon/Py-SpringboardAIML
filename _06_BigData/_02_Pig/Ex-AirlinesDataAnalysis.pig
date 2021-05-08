/*
- Find out top 20 airports by total volume of flights (Total traffic = Inbound + Outbound)
- We will list the popular carriers by (log base 10) of total flights for a 10 year span
 */

-- Load the Airlines Data
airlines_data = LOAD '/user/PigData/Airlines.txt' using PigStorage(',');

-- Once the data is loaded we will create two datasets grouping by Origin Airport and Destination Airport
-- The Count in each of the groups by Airport will give us the total traffic by airport - Outbound + Inbound
airportData_origin = GROUP airlines_data by $16;
airportData_destination = GROUP airlines_data by $17;

-- We will now compute total traffic by airport IATA code
airportData_origin_Traffic = FOREACH airportData_origin GENERATE group AS AirportCode, COUNT(airlines_data) AS TotalTraffic;
airportData_destination_Traffic = FOREACH airportData_destination GENERATE group AS AirportCode, COUNT(airlines_data) AS TotalTraffic;

-- Now getting traffic information for all Airports in the dataset
airportDataTraffic = UNION airportData_origin_Traffic, airportData_destination_Traffic;

-- To find the top 20 airports by traffic Volume
airportDataTrafficOrdered = RANK airportDataTraffic BY TotalTraffic DESC DENSE;
airportDataTrafficOrdered_Top20 = FILTER airportDataTrafficOrdered BY rank_airportDataTraffic <= 20;
--dump airportDataTrafficOrdered_Top20;

/*
We will list the popular carriers by (log base 10) of total flights for a 10 year span
 */
carrier_group = GROUP airlines_data by ($0, $8);
-- total flights by carrier by Year - using log of total traffic
carrier_totalFlights = FOREACH carrier_group GENERATE group, LOG(COUNT(airlines_data)) AS TotalFlights;
dump carrier_totalFlights;



