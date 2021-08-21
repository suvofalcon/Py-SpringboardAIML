"""
Geosales - Average unit_price by country for a given item type in a certain year
"""

# Library imports
from mrjob.job import MRJob
from mrjob.step import MRStep

# Define the main class which takes MRJob as an argument

class AverageUnitPrice(MRJob):

# Define the mapper and reducer functions
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_unitPrice,
                   reducer=self.reducer_avg_unitPrice)
        ]

    # Mapper function
    '''
    This function, will extract the Year from order_date , country, item type and unit price
    A composite key will be defined
    '''
    def mapper_get_unitPrice(self, _, line):
        # We will use the datetime library (only needed in mapper function)
        from datetime import datetime
        geosales = line.split(",")
        country, item_type, unit_price = geosales[2], geosales[3], float(geosales[10])
        # Extract the year from the order date column
        year = datetime.strptime(geosales[6], "%Y-%m-%d %H:%M:%S").year
        yield (year, item_type, country), unit_price
    
    # Reducer function
    '''
    Reducer function will sum the units, sold by the composite key
    '''
    def reducer_avg_unitPrice(self, key, value):
        # We will calculate the average using the length of the key as count
        # We will round of the price by 2 places of decimal
        yield key, round(sum(value)/len(key), 2)

if __name__ == '__main__':
    AverageUnitPrice.run()




