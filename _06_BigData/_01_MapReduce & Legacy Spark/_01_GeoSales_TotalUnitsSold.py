"""
Geosales - Total units_sold by year for a given country and a given item type
"""

# Library imports
from mrjob.job import MRJob
from mrjob.step import MRStep
from datetime import datetime

# Define the main class which takes MRJob as an argument

class TotalUnitsSold(MRJob):

# Define the mapper and reducer functions
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_unitsSold,
                   reducer=self.reducer_sum_unitsSold)
        ]

    # Mapper function
    '''
    This function, will extract the Year from order_date , country, item type and units sold
    A composite key will be defined
    '''
    def mapper_get_unitsSold(self, _, line):
        geosales = line.split(",")
        country, item_type, units_sold = geosales[2], geosales[3], int(geosales[9])
        year = datetime.strptime(geosales[6], "%Y-%m-%d %H:%M:%S").year
        yield (year, country, item_type), units_sold
    
    # Reducer function
    '''
    Reducer function will sum the units, sold by the composite key
    '''
    def reducer_sum_unitsSold(self, key, value):
        yield key, sum(value)

if __name__ == '__main__':
    TotalUnitsSold.run()




