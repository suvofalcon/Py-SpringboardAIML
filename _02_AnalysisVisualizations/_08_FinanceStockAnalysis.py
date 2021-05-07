"""
Finance Data Analysis
Banks Stocks data analysis during crisis period
We'll focus on bank stocks and see how they progressed throughout the financial crisis all the way to early 2016.
We need to get data using pandas datareader. We will get stock information for the following banks:

Bank of America
CitiGroup
Goldman Sachs
JPMorgan Chase
Morgan Stanley
Wells Fargo
@author: suvosmac
"""
from pandas_datareader import data, wb
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Lets set the time period during which we need to get the stock information
start = dt.datetime(2017, 1, 1)  # YYYY, MM, DD
end = dt.datetime(2018, 1, 1)

# Lets get the bank stock data
BAC = data.DataReader('BAC', 'iex', start=start, end=end, access_key="IEX_API_TOKEN")  # Bank of America
C = data.DataReader('C', 'google', start=start, end=end)  # CitiGroup
GS = data.DataReader('GS', 'google', start=start, end=end)  # Goldman Sachs
JPM = data.DataReader('JPM', 'google', start=start, end=end)  # JP Morgan Chase
MS = data.DataReader('MS', 'google', start=start, end=end)  # Morgan Stanley
WFC = data.DataReader('WFC', 'google', start, end)  # Wells Fargo

'''
# Could also do this for a Panel Object
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'], 'google', start, end)
'''
# Create a list of the ticker symbols (as strings) in alphabetical order.
# Call this list: tickers
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

'''
Use pd.concat to concatenate the bank dataframes together to a single data frame 
called bank_stocks. Set the keys argument equal to the tickers list.
'''
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
# Set the column name levels
bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

'''
Now we will do some Exploratory Data Analysis by leveraging multi level indexing
'''
# What is the max Close price for each bank's stock throughout the time period?
bank_stocks.xs(key='Close', axis=1, level='Stock Info').max()
# level is the name of the inner/outer data frame..
# here the inner data frame is stock info and outer dataframe is Bank Ticker

'''
Create a new empty DataFrame called returns. This dataframe will contain the returns 
for each bank's stock.
We can use pandas pct_change() method on the Close column to create a column 
representing this return value. Create a for loop that goes and for each 
Bank Stock Ticker creates this returns column and set's it as a column in the 
returns DataFrame.
'''
returns = pd.DataFrame()
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
# check the returns data frame
returns.head()

# Create a pairplot using seaborn of the returns dataframe.
sns.pairplot(returns[1:])  # ignoring the NaN values
plt.tight_layout()

# Using this returns DataFrame, figure out on what dates
# each bank stock had the best and worst single day returns.

# Best single day return for each bank stock
returns.idxmax()
# worst single day return for each bank stock
returns.idxmin()

# standard deviation for all the stocks
returns.std()
# If we have to see std for a specific period (note date is index now)
returns.ix['2016-01-01':'2016-12-31'].std()

# Create a distplot using seaborn during a period for Morgan Stanley
sns.distplot(returns.ix['2016-10-31':'2016-12-31']['MS Return'], color='green', bins=20)

# Create a line plot showing Close price for each bank for the entire index of time.
for tick in tickers:
    bank_stocks[tick]['Close'].plot(label=tick, figsize=(12,4))
plt.legend()
# We can do the same using cross section
bank_stocks.xs(key='Close', axis=1, level='Stock Info').plot()