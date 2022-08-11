"""
Demonstrate Resampling, Shifting, Rolling and Expanding of TimeSeries Data

@author: suvosmac
"""
# Time Resampling is aggregating timeseries data with some sort of frequencies

# Library imports
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset ( we will use the 'Date' Column as a named index)
# However in order for Pandas to read the Dates of index as DateTime objects and not string, we use one more
# additional parameter called parse_dates
df = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/starbucks.csv',
                 index_col='Date', parse_dates=True)
# Check the first few rows
print(df.head())

# Now if we check the indexes, we would see Pandas considers each data point as a datetime object
df.index

"""
When calling .resample() you first need to pass in a rule parameter, then you need to call some sort of aggregation 
function.

The rule parameter describes the frequency with which to apply the aggregation function (daily, monthly, yearly, etc.)
It is passed in using an "offset alias" - refer to the table below. [reference]

The aggregation function is needed because, due to resampling, we need some sort of mathematical rule to join the 
rows (mean, sum, count, etc.)

TIME SERIES OFFSET ALIASES

ALIAS	            DESCRIPTION
B	                business day frequency
C	                custom business day frequency (experimental)
D	                calendar day frequency
W	                weekly frequency
M	                month end frequency
SM	                semi-month end frequency (15th and end of month)
BM	                business month end frequency
CBM	                custom business month end frequency
MS	                month start frequency
SMS	                semi-month start frequency (1st and 15th)
BMS	                business month start frequency
CBMS	            custom business month start frequency
Q	                quarter end frequency
intentionally left blank
 
ALIAS	            DESCRIPTION
BQ	                business quarter endfrequency
QS	                quarter start frequency
BQS	                business quarter start frequency
A	                year end frequency
BA	                business year end frequency
AS	                year start frequency
BAS	                business year start frequency
BH	                business hour frequency
H	                hourly frequency
T,                  min	minutely frequency
S	                secondly frequency
L, ms	            milliseconds
U, us	            microseconds
N	                nanoseconds
"""
# resample this stock data of Starbucks
df.resample(rule='A').mean()  # we get average closing price by year with average volume

# We can use our own custom resampling function


def first_day(entry):
    """
    Returns the first instance of the period, regardless of sampling rate.
    """
    if len(entry):
        return entry[0]  # return the first entry


# This will return the first day closing price and first day volume for every single year
df.resample(rule='A').apply(first_day)

# we can use resample on a specific column
df['Close'].resample(rule='A').mean()
# We can straight away use pandas visualization here
df['Close'].resample(rule='A').mean().plot.bar(title='Yearly Mean Closing Price for Starbucks')
plt.show()

# Now we will do month wise resampling with max closing price and visualize the same
df['Close'].resample('M').max().plot.bar(title='Monthly Max Closing price for Starbucks')
plt.show()

"""
Sometimes we may need to shift all your data up or down along the time series index. 
"""
# Check the first five and last five rows
print(df.head)
print(df.tail)

'''
.shift() forward
This method shifts the entire date index a given number of rows, without regard for time periods (months & years).
It returns a modified copy of the original DataFrame.
'''
df.shift(1).head()
# NOTE: We will lose that last piece of data that no longer has an index!
df.shift(1).tail()

# The first data gets filled filled up as NaN - as it got shifted... If we dont want NaN and wants to fill it with
# zeros
df.shift(periods=1, fill_value=0.0).head()

'''
.shift() backwards
'''
df.shift(-1).head()  # Move everything back 1 - So we will technically loose the first row

'''
This shifting can also happen, using timeseries frequency codes
'''
df.shift(periods=1, freq='M').head()

"""
Rolling and Expanding

A common process with time series is to create data based off of a rolling mean.
The idea is to divide the data into "windows" of time, and then calculate an aggregate function for each window. 
In this way we obtain a simple moving average. 
"""

# Lets plot the closing price
df['Close'].plot(figsize=(12, 5))
plt.show()

# We will now attempt to create a rolling mean for seven days
df.rolling(window=7).mean()
# The first 6 values will become NaN, because it wont have enough data to calculate
# an average for these days, but from the seventh day onwards, we will have values

# Now we will attempt to plot close price with a 30 day rolling mean
df['Close'].plot(figsize=(12, 5))
df.rolling(window=30).mean()['Close'].plot()
plt.show()

# If we reduce the window size, it will become closer to fitting the data
df['Close'].plot(figsize=(12, 5), title="Plot of Closing vs 15 Days Moving Average")
df.rolling(window=15).mean()['Close'].plot()
plt.show()

# To add a new column to the original dataframe
df['Close: 15 Day Mean'] = df['Close'].rolling(window=15).mean()
df[['Close', 'Close: 15 Day Mean']].plot(figsize=(12, 5),
                                         title="Plot of Closing vs 15 Days Moving Average")
plt.show()

"""
Expanding
Instead of calculating values for a rolling window of dates, what if you wanted to take into account everything from 
the start of the time series up to each point in time? For example, instead of considering the average over the 
last 7 days, we would consider all prior data in our expanding set of averages.
"""
df['Close'].expanding(min_periods=15).mean().plot(figsize=(12, 5))
plt.show()

# So essentially the mean would keep on increasing and from 15th day onwards the average value will stabilise,
# This will show not much volatality in stocks...


