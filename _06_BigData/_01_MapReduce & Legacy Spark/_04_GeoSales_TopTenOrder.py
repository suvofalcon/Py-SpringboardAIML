
"""
Geosales - Top 10 order id for a given year by the total_profit 

"""
from mrjob.job import MRJob
from mrjob.step import MRStep
from heapq import nlargest

MRJob.SORT_VALUES = True

class TopTenOrder(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_order_by_profit,
                   combiner=self.combiner_arrange_by_profits,
                   reducer=self.reducer_reduce_by_profits),
            MRStep(reducer=self.reducer_sorted_by_topTen_profits)
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


    # Combiner Function
    def combiner_arrange_by_profits(self, key, values):
        # We are padding the values with 0 so that the string sorting happens correct
        yield str(values).zfill(5), key 
        
    # Reducer Function
    def reducer_reduce_by_profits(self, key, profits):
        yield key, profits
    
    def top_10(self, orderPair):
        for orderTotal, itemID in orderPair:
            top_orders = nlargest(10, orderTotal)
        for top_orders in orderTotal:
            return (orderTotal, itemID)
    
    # Reducer Function
    def reducer_find_top10(self, key, orderPair):
        orderTotal, country = self.top_10(orderPair)
        yield orderTotal, country

           
if __name__ == '__main__':
    TopTenOrder.run()