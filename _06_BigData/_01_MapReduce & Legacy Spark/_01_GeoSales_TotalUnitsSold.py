"""
Geosales - Total units_sold by year for a given country and a given item type
"""

# Library imports
from mrjob.job import MRJob
from mrjob.step import MRStep

# Define the main class which takes MRJob as an argument

class TotalUnitsSold(MRJob):

# Define the mapper and reducer functions
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_unitsSold,
                   reducer=self.reducer_sum_unitsSold)
        ]

    # Mapper function
    def mapper_get_unitsSold(self, _, line):
        geosales = line.split(",")
        yield (geosales[2], geosales[3]), int(geosales[9])
    
    # Reducer function
    def reducer_sum_unitsSold(self, key, value):
        yield key, sum(value)

if __name__ == '__main__':
    TotalUnitsSold.run()




