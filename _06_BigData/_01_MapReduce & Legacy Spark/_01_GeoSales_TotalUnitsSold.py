"""
Geosales - Total Units Sold by Country
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
        (index, region, country, item_type, sales_channel, order_priority, order_date, order_id, 
        ship_date, units_sold, unit_price, unit_cost, total_revenue, total_cost, total_profit) = line.split(",")

        yield country, units_sold

    # Reducer function
    def reducer_sum_unitsSold(self, key, values):
        yield str(sum(values)).zfill(5), key 

if __name__ == '__main__':
    TotalUnitsSold.run()




