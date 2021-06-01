# Get the Spark libraries

from pyspark import SparkConf, SparkContext
import collections
import re

# Create the Spark Context , set Application NAme
conf = SparkConf().setMaster("local").setAppName("CountWordOccurrences")
sc = SparkContext(conf=conf)

'''
This function will normalize the words, take care of all punctuations and convery every word to lowercase
We will use regular expressions to achieve this
'''
def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

def printWordCounts(wordDictionary):
    for word, count in wordDictionary.items():
        cleanWord = word.encode('ascii', 'ignore')
        if (cleanWord):
            print(cleanWord.decode() + " " + str(count))

# Read the data file
file = sc.textFile("../resources/Book.txt")

# We will read every word of the file, using flatmap
words = file.flatMap(normalizeWords)

'''
# Now lets call the function to print wordCount
wordCounts = words.countByValue()
printWordCounts(wordCounts)
'''

# Now if we want to print things sorted
'''
The below will return following sample
('offers', 16), ('intentionally', 2), ('limited', 12), ('expense', 10)
'''
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

# Now we need to sort based on the number of times the words have occured.. We would flip this around
# So our keys become values and values becomes keys
wordCountsSorted = wordCounts.map(lambda x : (x[1], x[0])).sortByKey()
results = wordCountsSorted.collect()

# we will print the results
for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count)

# # We will print the dictionary
# for key, value in wordCount.items():
#     cleanWord = key.encode('ascii', 'ignore')
#     if (cleanWord):
#         print(cleanWord.decode() +" "+ str(value))

# # Now we will print the dictionary
# for word, count in wordCount.items():
#     cleanWord = word.encode('ascii', 'ignore')
#     if(cleanWord):
#         print cleanWord, count