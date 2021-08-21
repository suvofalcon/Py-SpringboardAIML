"""
Geosales - Top 10 order id for a given year by the total_profit 
"""

# Library imports
from mrjob.job import MRJob
from mrjob.step import MRStep

class TopOrderByProfit(MRJob):

    # Define the mapper and reducer functions
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_profits,
                   combiner=self.combiner_combine_profits,
                   reducer=self.sort_by_profits)
        ]

    ## Code goes here





if __name__ == '__main__':
    TopOrderByProfit.run()
