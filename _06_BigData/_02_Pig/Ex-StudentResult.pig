/* FilterDemo.pig
In this task We have 2 files named as student and results in the folder.
Student file: contains names and roll number of students.
Results file: contains roll number and results of students whether they passed or failed.
We need to print the name of all the students who failed the exam.
*/

-- Load the student file
students = LOAD '/user/PigData/Student' AS (studentName:chararray, rollNumbers:int);
-- Load the result file
results = LOAD '/user/PigData/Result' AS (rollNumbers:int, resultStatus:chararray);

-- We will join the datasets using roll numbers
joined_dataset = JOIN students BY rollNumbers, results BY rollNumbers;

-- We will now filter the data based on resultStatus == 'fail'
failed_students = FILTER joined_dataset BY resultStatus == 'fail';
failed_students_name = FOREACH failed_students GENERATE studentName as failedNames;
dump failed_students_name;
