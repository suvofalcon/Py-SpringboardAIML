"""
Exercise on a monthly milk production dataset

Exercise on Value of Manufacturers' Shipments for All Manufacturing Industries from
UMTMVS Dataset - https://fred.stlouisfed.org/series/UMTMVS
"""
# Library imports
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/monthly_milk_production.csv",
                 encoding='utf-8')
title = "Monthly Milk Production: pounds per cow, Jan '62 - Dec ' 75"

# Check the dataframe
print(df.head())
print(df.info())

# 1. What is the current data type of the Date column?
print(df.dtypes)
print(df['Date'].dtypes)

# 2. Change the Date column to a datetime format
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].dtypes)

# 3. Set the Date column to be the new index
df.set_index('Date', inplace=True)
df.head()

# 4. Plot the DataFrame with a simple line plot. What do you notice about the plot?
# Milk production is increasing year on year. Shows consistent seasonality and as well as upward trend
df.plot()
plt.show()

# 5. Add a column called 'Month' that takes the month value from the index
df['Month'] = df.index.month
df.head()
# Another solution
df['Month'] = df.index.strftime('%B')
df.head()

# 6. Create a BoxPlot that groups by the Month field
ax = df.boxplot(by='Month', figsize=(12, 8))
ax.set(xlabel='Month', ylabel='Production Volume')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.title("Boxplot - Production by Month")
plt.suptitle('')
plt.show()

# Read in the data UMTMVS.csv file from the Data folder
umt_df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/UMTMVS.csv")
# Check the data load
print(umt_df.head())
print(umt_df.info())

# Set the DATE column as the index.
umt_df.set_index('DATE', inplace=True)
print(umt_df.head())

# Check the data type of the index.
print(umt_df.index)

# Convert the index to be a datetime index.
umt_df.index = pd.to_datetime(umt_df.index)
print(umt_df.index)

# We will plot the data
umt_df.plot(figsize=(14, 8))
plt.show()

# What was the percent increase in value from Jan 2009 to Jan 2019?
100 * (umt_df.loc['2019-01-01'] - umt_df.loc['2009-01-01'])/umt_df.loc['2009-01-01']

# What was the percent decrease from Jan 2008 to Jan 2009?
100 * (umt_df.loc['2009-01-01'] - umt_df.loc['2008-01-01'])/umt_df.loc['2008-01-01']

# What is the month with the least value after 2005?
umt_df.loc['2005-01-01':].idxmin()

# What 6 months have the highest value?
umt_df.sort_values(by='UMTMVS', ascending=False).head(5)

# How many millions of dollars in value was lost in 2008?
# (Another way of posing this question is what was the value difference between Jan 2008 and Jan 2009)
umt_df.loc['2008-01-01'] - umt_df.loc['2009-01-01']

# Create a bar plot showing the average value in millions of dollars per year
umt_df.resample(rule='Y').mean().plot.bar(figsize=(15, 8))
plt.show()

# What year had the biggest increase in mean value from the previous year's mean value?
yearly_data = umt_df.resample(rule='Y').mean()
yearly_data_shift = yearly_data.shift(1)
change = yearly_data - yearly_data_shift
change['UMTMVS'].idxmax()

# Plot out the yearly rolling mean on top of the original data. Recall that this is monthly data and there are
# 12 months in a year!
umt_df['Yearly Mean'] = umt_df['UMTMVS'].rolling(window=12).mean()
umt_df[['UMTMVS', 'Yearly Mean']].plot(figsize=(12, 5)).autoscale(axis='x', tight=True)
plt.show()

# Some month in 2008 the value peaked for that year. How many months did it take to surpass that 2008 peak?
# (Since it crashed immediately after this peak)
umt_df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/TimeSeriesData/UMTMVS.csv",
                     index_col='DATE', parse_dates=True)
umt_df.head()

# We will extract a dataframe with 2008 values
umt_df2008 = umt_df.loc['2008-01-01': '2009-01-01']
# Check the date on which the max volume happened
umt_df2008.idxmax()
# Check the max volume
umt_df2008.max()

# So we will extract all the values post peak
umt_df_postPeak = umt_df['2008-06-01':]
# subsetting all the entries which is larger than the
umt_df_postPeak[umt_df_postPeak >= umt_df2008.max()].dropna()

# So we see from the first entry after the peak in 2008-06-01, the month, in which it crossed the value
# is only in 2014-03-01
len(umt_df.loc['2008-06-01':'2014-03-01'])