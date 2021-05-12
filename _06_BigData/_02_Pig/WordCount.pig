/* WordCount.pig
This script counts the number of words from an input file in hdfs
*/

-- Load the data from the hdfs path. The data is loaded into 'lines' variable where each line is a chararray
-- LOAD is the operator to load the data from hdfs
-- 'lines' is the relation that data is loaded into and is a bag of tuples, each tuple from the bag is one record from the file
lines = LOAD '/datasets/wcinput' AS (line:chararray);

-- If we want to see what is there in the relation 'lines', we use the following command
--describe lines; -- this will show the schema of the relation, which is what will pig expect to see, when the data is loaded

-- In this code, we will split each sentence into words
words = FOREACH lines GENERATE FLATTEN(TOKENIZE (line)) as word;
-- We will group the words based on the word itself
grouped = GROUP words by word;
-- Then we will count the number of times each word occurs
wordcount = FOREACH grouped GENERATE group, COUNT(words);

-- This will execute the script and dump the output on the screen
--dump wordcount;
STORE wordcount INTO '/datasets/wcoutput';
