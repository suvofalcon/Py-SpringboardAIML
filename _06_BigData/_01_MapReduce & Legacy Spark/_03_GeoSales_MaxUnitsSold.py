"""
Geosales - Max and Min units sold in any order for each year by country for a given item type. 
"""

# Library imports
from mrjob.job import MRJob
from mrjob.step import MRStep

class MaxUnitsSold(MRJob):

    # Define the mapper and reducer functions
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_units,
                   reducer=self.reducer_max_units_by_keys)
        ]

    # Mapper function
    '''
    This function, will extract the Year from order_date , country, item type and units sold
    A composite key will be defined
    '''
    def mapper_get_units(self, _, line):
        from datetime import datetime
        geosales = line.split(",")
        # Extract the order id and total profit
        country, item_type, units_sold = geosales[2], geosales[3], int(geosales[9])
        # Extract the year from the order date column
        year = datetime.strptime(geosales[6], "%Y-%m-%d %H:%M:%S").year
        yield (year, country, item_type), units_sold
    
    # Reducer function
    '''
    Reducer function will return the max units, sold by the composite key
    '''
    def reducer_max_units_by_keys(self, key, value):
        yield key, max(value)
        
if __name__ == '__main__':
    MaxUnitsSold.run()
