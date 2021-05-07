"""
Demonstrate python Datetime object, numpy datetime and Pandas datetime

@author: suvosmac
"""

# Library imports
from datetime import datetime
import numpy as np
import pandas as pd

# create few variables
my_year = 2020
my_month = 1
my_day = 2
my_hour = 13
my_min = 30
my_sec = 15

'''
Python built in datetime object
'''
# create a datetime object
my_date = datetime(my_year, my_month, my_day)
print(my_date)

my_date_time = datetime(my_year, my_month, my_day, my_hour, my_min, my_sec)
print(my_date_time)

# We can use attributes to extract information out of the datetime object
print(my_date_time.hour)

'''
Using datetime object of numpy
'''
# If we need to create an array of datetime objects
np.array(['2020-03-15', '2020-03-16', '2020-03-17'], dtype='datetime64')  # Default day level precision
np.array(['2020-03-15', '2020-03-16', '2020-03-17'], dtype='datetime64[Y]')  # Year level precision
np.array(['2020-03-15', '2020-03-16', '2020-03-17'], dtype='datetime64[M]')  # Month level precision

# we can use np.arange for datetime objects as well
np.arange('2018-06-01', '2018-06-23', 7, dtype='datetime64[D]')
np.arange('1968', '1976', dtype='datetime64[Y]')

'''
Pandas datetime object
'''
# create datetime object
pd.date_range('2020-01-01', periods=7, freq='D')
pd.date_range('Jan 01, 2018', periods=7, freq='D')
# There are variety of string formats which Pandas can take and understand it as a date. But all that has to be
# one within that predefined lists

# using the pandas built in datetime function
pd.to_datetime(['1/2/2018', 'Jan 03, 2018'])
pd.to_datetime(['1/2/2018', '1/3/2018'], format='%d/%m/%Y')  # mention our own date format
pd.to_datetime(['2--1--2018', '3--1--2018'], format='%d--%m--%Y')

# Lets create some random date
data = np.random.rand(3, 2)
cols = ['A', 'B']
print(data)
idx = pd.date_range('2020-01-01', periods=3, freq='D')
df = pd.DataFrame(data, index=idx, columns=cols)
print(df)

print(df.index)
print(df.index.max())  # oldest date value
print(df.index.argmin())  # location of the earliest date value
