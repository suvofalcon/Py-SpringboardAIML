
"""
Geosales - Top 10 order id for a given year by the total_profit 

"""

from typing import OrderedDict
from mrjob.job import MRJob
from mrjob.step import MRStep

MRJob.SORT_VALUES = True

class TopTenOrder(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_order_by_profit,
                   reducer=self.reducer_arrange_by_profits),
            MRStep(reducer=self.reducer_sorted_by_profits)
        ]


    # Mapper Function
    def mapper_get_order_by_profit(self, _, line):
        from datetime import datetime
        # Split the data
        geosales = line.split(',')
        # Extract the order and profit
        order_id, profit = geosales[7], geosales[14]
        # Extract the year from the order date column
        year = datetime.strptime(geosales[6], "%Y-%m-%d %H:%M:%S").year
        yield (year, order_id), float(profit)


    # Reducer Function
    def reducer_arrange_by_profits(self, key, values):
        # We are padding the values with 0 so that the string sorting happens correct
        yield str(values).zfill(5), key 
        #yield key, sum(values)
    

    # Reducer Function
    def reducer_sorted_by_profits(self, key, profits):
        yield key, profits
           
if __name__ == '__main__':
    TopTenOrder.run()