"""
Demonstrating Visualizations of TimeSeries Data

@author: suvosmac
"""
# Library imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates

# Read the dataset
df = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/starbucks.csv',
                 index_col='Date', parse_dates=True)

# Check the first few rows and dataset details
print(df.head())
print(df.info())

# Plotting the Close and Volume column
df['Close'].plot(title="Closing Prices for Stocks")
plt.show()
df['Volume'].plot(title='Volume of Stocks traded')
plt.show()

# If we want to give our own title, xlabel and ylabel and autoscale on the axis
title = 'Closing Prices for Stocks'
ylabel = 'Closing Prices'
xlabel = 'Dates'
ax = df['Close'].plot(title=title)
ax.autoscale(axis='both', tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

# There a way to specify a specific span of time on the X-Axis label. This can be done either on the
# dataframe or on the plot itself

# To plot the 2017 year data (Extracting the needed data from dataframe and then plot)
df['Close']['2017-01-01':'2017-12-31'].plot(figsize=(12, 5), title='2017 Stock closing prices')
plt.show()

# Now we can also alternately do that on the plot itself
df['Close'].plot(figsize=(12, 5), title='2017 Stock closing prices', xlim=['2017-01-01', '2017-12-31'])
plt.show()

# On a side by side they look different, because of a different scale on the Y-Axis
# We can specify Y-limits for both and see the closing price trend - For ex
# also add a linestyle
df['Close']['2017-01-01':'2017-12-31'].plot(figsize=(12, 5), title='2017 Stock closing prices',
                                            ylim=[30, 70], ls='--', color='red')
plt.show()

# how to change the format and appearance of dates along the x-axis.


"""
# Lets say we want to see the closing prices for three months in 2017

Notice that dates are spaced one week apart. The dates themselves correspond with byweekday=0, or Mondays.
For a full list of locator options available from matplotlib.dates 
visit https://matplotlib.org/api/dates_api.html#date-tickers

There can also be a way to do different date formatting on the plots itself

CODE	                MEANING	                                                    EXAMPLE
%Y	                    Year with century as a decimal number.	                    2001
%y	                    Year without century as a zero-padded decimal number.	    01
%m	                    Month as a zero-padded decimal number.	                    02
%B	                    Month as locale’s full name.	                            February
%b	                    Month as locale’s abbreviated name.	                        Feb
%d	                    Day of the month as a zero-padded decimal number.	        03
%A	                    Weekday as locale’s full name.	                            Saturday
%a	                    Weekday as locale’s abbreviated name.	                    Sat
%H	                    Hour (24-hour clock) as a zero-padded decimal number.	    16
%I	                    Hour (12-hour clock) as a zero-padded decimal number.	    04
%p	                    Locale’s equivalent of either AM or PM.	                    PM
%M	                    Minute as a zero-padded decimal number.	                    05
%S	                    Second as a zero-padded decimal number.	                    06
%#m	                    Month as a decimal number. (Windows)	                    2
%-m	                    Month as a decimal number. (Mac/Linux)	                    2
%#x	                    Long date	                                                Saturday, February 03, 2001
%#c	                    Long date and time	                                        Saturday, February 03, 2001 16:05:06

"""
ax = df['Close'].plot(figsize=(12, 5), xlim=['2017-01-01', '2017-03-01'], ylim=[50, 60])
ax.set(xlabel='')
# Now xaxis is more evenly spaced and start of the weekday is considered as monday (byweekday=0)
ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=0))
# We can also use date formatter
ax.xaxis.set_major_formatter(dates.DateFormatter('%a-%B-%d'))

# We can also use minor formatting
ax.xaxis.set_minor_locator(dates.MonthLocator())
ax.xaxis.set_minor_formatter(dates.DateFormatter('\n\n%b'))  # \n for new line for better readability

# Finally we will show the gridlines
ax.xaxis.grid(True)
ax.yaxis.grid(True)

plt.show()
