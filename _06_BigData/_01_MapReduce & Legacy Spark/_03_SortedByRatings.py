
"""
Sort the movies by their numbers of ratings

- Map to (MovieID, 1) - Key value pairs
- Reduce with output of (rating count, movieID)
- Send this to second reducer so we end up with things sorted by rating count

"""

from mrjob.job import MRJob
from mrjob.step import MRStep

class RatingsBreakdown(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                   reducer=self.reducer_count_ratings),
            MRStep(reducer=self.reducer_sorted_output)
        ]


    # Mapper Function
    def mapper_get_ratings(self, _, line):
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield movieID, 1


    # Reducer Function
    def reducer_count_ratings(self, key, values):
        # We are padding the values with 0 so that the string sorting happens correct
        yield str(sum(values)).zfill(5), key 


    # Reducer Function
    def reducer_sorted_output(self, count, movies):
        for movie in movies:
                yield movie, count


if __name__ == '__main__':
    RatingsBreakdown.run()
