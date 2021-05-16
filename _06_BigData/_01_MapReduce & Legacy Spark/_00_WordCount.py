"""
Count the number of words in the file
"""

from mrjob.job import MRJob

class WordCount(MRJob):
    
    '''
    The below mapper() function defines the mapper for MapReduce and takes 
    key value argument and generates the output in tuple format . 
    The mapper below is splitting the line and generating a word with its own 
    count i.e. 1
    '''
    def mapper(self, _, line):
        for word in line.split():
            yield(word, 1)
    
    '''
    The below reducer() is aggregating the result according to their key and
    producing the output in a key-value format with its total count
    '''
    def reducer(self, word, counts):
        yield(word, sum(counts))
    

# We will initiate the Map-Reduce job
if __name__ == '__main__':
    WordCount.run()
