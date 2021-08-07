"""
Sort Movies by Popularity with Hadoop
"""

from mrjob.job import MRJob
from mrjob.step import MRStep

class RatingsBreakdown(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                   reducer=self.reducer_count_ratings)
        ]

    # Mapper Function    
    def mapper_get_ratings(self, _, line):
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield movieID, 1

    
    # Reducer Function    
    def reducer_count_ratings(self, key, values):
        yield str(sum(values)).zfill(5), key 

if __name__ == '__main__':
    RatingsBreakdown.run()