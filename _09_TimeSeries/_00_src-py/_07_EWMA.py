"""
Exponentially Weighted moving averages

We just showed how to calculate the SMA based on some window. However, basic SMA has some weaknesses:
- Smaller windows will lead to more noise, rather than signal
- It will always lag by the size of the window
- It will never reach to full peak or valley of the data due to the averaging.
- Does not really inform you about possible future behavior, all it really does is describe trends in your data.
- Extreme historical values can skew your SMA significantly
To help fix some of these issues, we can use an EWMA (Exponentially weighted moving average).


@author: suvosmac
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
airline = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/airline_passengers.csv",
                      index_col='Month')
# remove all missing values
airline.dropna()

# change the index to datetime
airline.index = pd.to_datetime(airline.index)

# Now lets check the initial rows of the dataframe
print(airline.head())

# Now lets create some Simple Moving averages (6 months, 12 months etc)
airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()

# Check the head
print(airline.head())
# Now lets plot the data
airline.plot()
plt.show()

# Now we will implement the EWMA
airline['EWMA-12'] = airline['Thousands of Passengers'].ewm(span=12).mean()
# Now we will again plot the data but, this time will plot the actuals, SMA-12 and EWMA-12
airline[['Thousands of Passengers', '12-month-SMA', 'EWMA-12']].plot()
plt.show()
# We see that the for EWMA-12 the trend is flat in the beginning and towards the recent time, the seasonality
# starts to show up, because the recent points are weighted more than the older data points.

