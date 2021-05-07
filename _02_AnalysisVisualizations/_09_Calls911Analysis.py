"""
Analysis and Visualizations on a dataset which lists various calls made to 911
For this capstone project we will be analyzing some 911 call data from Kaggle. The data contains the following fields:

lat : String variable, Latitude
lng: String variable, Longitude
desc: String variable, Description of the Emergency Call
zip: String variable, zipcode
title: String variable, Title
timeStamp: String variable, YYYY-MM-DD HH:MM:SS
twp: String variable, Township
addr: String variable, Address
e: String variable, Dummy variable (always 1)

@author: suvosmac
"""
# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
calls_df = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/911.csv')
# Check the data load - load the first three rows
print(calls_df.head(3))

# Check the detailed data frame structure
print(calls_df.info())

# What are the top 5 zip codes for 911 calls
calls_df['zip'].value_counts().head(5)

# What are the top 5 townships (twp) for 911 calls
calls_df['twp'].value_counts().head(5)

# How many unique title codes are there
calls_df['title'].nunique()

'''
In the titles column there are "Reasons/Departments" specified before the title code. 
These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column 
called "Reason" that contains this string value.
For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.
'''
calls_df['Reason'] = calls_df['title'].apply(lambda string:string.split(":")[0])
# What is the most common reason for 911 calls based on this column
calls_df['Reason'].value_counts().head(5)

# To visualize this using Seaborn, countplot of 911 calls by reason
sns.countplot(x='Reason', data=calls_df)
plt.show()

# What is the data type of the objects in the timeStamp column?
type(calls_df['timeStamp'][0])  # the type is string
# convert it to DateTime objects
calls_df['timeStamp'] = pd.to_datetime(calls_df['timeStamp'])

# Now we will extract month, day of week and hour from this attribute and create three columns
calls_df['Hour'] = calls_df['timeStamp'].apply(lambda timeStamp:timeStamp.hour)
calls_df['Month'] = calls_df['timeStamp'].apply(lambda timeStamp:timeStamp.month)
calls_df['DayOfWeek'] = calls_df['timeStamp'].apply(lambda timeStamp:timeStamp.dayofweek)

# we will now map the actual string names to the day of the week
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: "Fri", 5: "Sat", 6: "Sun"}
# to map it in the dataframe
calls_df['DayOfWeek'] = calls_df['DayOfWeek'].map(dmap)

# Now creating a countplot for the DayOfWeek based on the Reason Column (by ReasonColumn)
sns.countplot(x='DayOfWeek', data=calls_df, hue='Reason')
# To move the legend out
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

# Let us run the same for month
sns.countplot(x='Month', data=calls_df, hue='Reason')
# To move the legend out
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
# Some months like 9,10 and 11 are missing
plt.show()

# We would represent it in a different manner.. aggregating the dataframe by month on count
byMonth = calls_df.groupby('Month').count()
# If we just draw a line plot of the lat column


# try to plot a linear fit on the number of calls per month.
# to plot a linear fit, in this case we have to reset the index to a column
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
plt.show()
# If we have to linearly approximate the relationship between the number of calls to 911 from the
# townships, we can conclude that month on month the number of calls is decreasing

# Create a new column called 'Date' that contains the date from the timeStamp column.
calls_df['Date'] = calls_df['timeStamp'].apply(lambda timeStamp:timeStamp.date())
# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
calls_df.groupby('Date').count()['lat'].plot()
plt.show()
plt.tight_layout()

# Now recreate this plot but create 3 separate plots with each plot
# representing a Reason for the 911 call
plt.subplot(1, 3, 1)
calls_df[calls_df['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.subplot(1, 3, 2)
calls_df[calls_df['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.subplot(1, 3, 3)
calls_df[calls_df['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.show()

'''
Now we will create heatmaps with seaborn and data.. To create heatmaps, we need to convert
our data into matrix format
With heatmap we want to understand how is the variation between the day of the week 
and hour of the display
'''

# Lets first do an aggregate by count by DayOfWeek by Hour
calls_df.groupby(by=['DayOfWeek', 'Hour']).count()
# examine by a single column, which is more complete
calls_df.groupby(by=['DayOfWeek', 'Hour']).count()['lat']
# now to convert the above into a matrix form
dayHourMatrix = calls_df.groupby(by=['DayOfWeek', 'Hour']).count()['lat'].unstack()
# now draw the heatmap
sns.heatmap(dayHourMatrix, cmap='viridis')
plt.show()
# draw the clustermap
sns.clustermap(dayHourMatrix)
plt.show()

# repeat these same plots and operations, for a DataFrame that shows the Month as the column.
calls_df.groupby(by=['DayOfWeek', 'Month']).count()
# examine by a single column, which is more complete
calls_df.groupby(by=['DayOfWeek', 'Month']).count()['lat']
# now to convert the above into a matrix form
dayMonthMatrix = calls_df.groupby(by=['DayOfWeek', 'Month']).count()['lat'].unstack()
# now draw the heat map
sns.heatmap(dayMonthMatrix, cmap='viridis')
plt.show()
# draw the clustermap
sns.clustermap(dayMonthMatrix)
plt.show()