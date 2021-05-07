"""
Pandas stands for Panel data library.
It is the most popular data analysis library for Python and comes with many tools

Demonstrate some key features of Pandas

@author : suvosmac
"""

import numpy as np
import pandas as pd
from numpy.random import randn
from sqlalchemy import create_engine

# lets create a list
labels = ['a', 'b', 'c']
mylist = [10, 20, 30]

arr = np.array(mylist)

# create a dictionary
d = {'a': 10, 'b': 20, 'c': 30}

# A series is a named index array
print(pd.Series(data=mylist))

print(pd.Series(data=arr, index=labels))

# A Series can also hold data of different types
print(pd.Series(data=[10, 'a', 4.1]))

ser1 = pd.Series(data=[1, 2, 3, 4], index=['USA', 'GERMANY', 'USSR', 'JAPAN'])
print(ser1)

# Now we can reference an elenent of the series with respect to index
print(ser1['USA'])

ser2 = pd.Series([1, 4, 5, 6], index=['USA', 'GERMANY', 'ITALY', 'JAPAN'])
print(ser2)

# Pandas can have operations on series, which will be based on index
print(ser1 + ser2)

'''

Pandas DataFrames
A Pandas dataframes is implemented as multiple series that share the same index.
It is essentially a tabular data storage format

'''
np.random.seed(101)
rand_mat = randn(5, 4)
print(rand_mat)

# create a dataframe
df = pd.DataFrame(data=rand_mat, index='A B C D E'.split(), columns='W X Y Z'.split())
print(df)
# reference a column
print(df['W'])
# every column / row is a pandas series
print(type(df['W']))

# reference multiple columns (for that we need to pass a list)
print(df[['W', 'Y']])

# create a new column
df['NEW'] = df['W'] + df['Y']
print(df)

# To drop this column
df.drop('NEW', axis=1, inplace=True)
print(df)

# We can also use drop for rows
print(df.drop('A'))

# There are two ways to reference/select rows
print(df.loc['A'])
print(df.iloc[0])  # Index location - index of the first row
# similarly selecting multiple rows
print(df.loc[['A', 'E']])
print(df.iloc[[0, 3]])

# Selecting a subset / slice
print(df.loc[['A', 'B'], ['Y', 'Z']])

# Use the ix property to slice a data frame upto 30 rows
df.ix[0:30]  # this slicing is based on indexes

# conditional selection from dataframes
print(df[df > 0])
print(df[df['W'] > 0])  # returns the data frame with rows where W > 0
# Further subset only 'Z' column
print(df[df['W'] > 0]['Z'])

# Further subset only Z column and E row
print(df[df['W'] > 0]['Z'].loc['E'])

# combining multiple conditions
print(df[(df['W'] > 0) & (df['Y'] > 1)])

df.reset_index()  # This converts the index to a column and adds a default

new_values = 'CA NY WY OR CO'.split()
df['States'] = new_values
print(df)
# Now if we need to use this states column as index
df.set_index('States', inplace=True)
print(df)

# to check the data types of the columns
print(df.dtypes)
# to check a short info on the dataframe
print(df.info())
# to check summary statistics of a dataframe
print(df.describe())

# To check distribution of a column based on certain condition
print((df['W'] > 0).value_counts())
# if we do sum, it will only return the number of True
print(sum(df['W'] > 0))

'''
Handle missing data with Pandas
'''
# Lets create a dataframe with some missing values and pass a dictionary
df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [5, np.nan, np.nan], 'C': [1, 2, 3]})
print(df)

# in order to drop all missing data - by default this happens row wise
df.dropna()
# in order to drop missing data column wise
df.dropna(axis=1)

# in order to drop missing values with threshold - For example, we will drop only when the
# number of missing values is more than 1
df.dropna(thresh=2)

# Fill in missing data
df.fillna(value=0)
# to fill with mean
df.fillna(value=df.mean())

# Some specific ways to fill in column wise
df['A'].fillna(value=df['A'].mean())

'''
Group By Operations

Group By operations involve
Split
Apply
Combine
'''

# We will fist create a dataframe
# create a dataframe
data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'], 'Person': ['Sam', 'Charlie', 'Amy',
                                                                            'Vanessa', 'Carl', 'Sarah'],
        'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)
print(df)

# By Company we want to find mean of sales
df.groupby('Company').mean()

# more operations

df.groupby('Company').std()
df.groupby('Company').min()
df.groupby('Company').max()
df.groupby('Company').count()

# See summary statistics by Company
df.groupby('Company').describe()
df.groupby('Company').describe().transpose()  # transform rows to columns and vice versa

# to see just for one company
df.groupby('Company').describe().transpose()['GOOG']

'''
Merging , Joining and Concatenating

'''
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])
print(df1)
print(df2)
print(df3)

'''
Concatenation
Concatenation basically glues together DataFrames. Keep in mind that dimensions should match along the axis you 
are concatenating on. You can use pd.concat and pass in a list of DataFrames to concatenate together:
'''

pd.concat([df1, df2, df3])  # default is column wise concatenation
pd.concat([df1, df2, df3], axis=1)  # row wise concatenation

# Example Dataframes

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
print(right)

'''
Merging
The merge function allows you to merge DataFrames together using a similar logic as merging SQL Tables together.
'''
pd.merge(left, right, how='inner', on='key')

# A more complicated example
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

# Only the matching combinations will be shown
pd.merge(left, right, on=['key1', 'key2'])
# all combinations will be shown
pd.merge(left, right, how='outer', on=['key1', 'key2'])
# All right(second) dataframe combinations
pd.merge(left, right, how='right', on=['key1', 'key2'])
# All left(first) dataframe combinations
pd.merge(left, right, how='left', on=['key1', 'key2'])

'''
Joining
Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames 
into a single result DataFrame.
'''
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

left.join(right)  # All indexes of left
left.join(right, how='outer')

'''
Other Common operations
'''
df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [444, 555, 666, 444], 'col3': ['abc', 'def',
                                                                               'ghi', 'xyz']})
print(df)

# Unique values in a column
df['col2'].unique()
# number of unique values
df['col2'].nunique()
# number of instances per unique values
df['col2'].value_counts()

# values where col1 > 2
# col2 == 444
df[(df['col1'] > 2) & (df['col2'] == 444)]

# create a simple function


def times_two(number):
    return number * 2

# we can apply this function to a column


df['col1'].apply(times_two)
df['new'] = df['col1'].apply(times_two)  # we can use this to add a new column
print(df)

# permanently remove a column
del df['new']
print(df)

# to see all the columns
print(df.columns)

# quick info about dataframe
print(df.info())

# to get the indexes
print(df.index)

# summary statistics
print(df.describe())

# sort and order a dataframe - default is ascending.
# Please note indexes are going to remain the same.
df.sort_values(by='col2')
df.sort_values(by='col2', ascending=False)  # This is descending

'''
Data Input and Output with Pandas
'''
df = pd.read_csv('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/example.csv')
print(df)
# to create a new dataframe
newdf = df[['a', 'b']]
print(newdf)

newdf.to_csv('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/mynew.csv',
             index=False)

# Pandas can read tables from html as well
mylist_of_tables = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(type(mylist_of_tables))  # The output is a list
len(mylist_of_tables)

# create the dataframe from this list
df = mylist_of_tables[0]
print(df)

'''
SQL

The pandas.io.sql module provides a collection of query wrappers to both facilitate data retrieval and to 
reduce dependency on DB-specific API. Database abstraction is provided by SQLAlchemy if installed. In addition you 
will need a driver library for your database. Examples of such drivers are psycopg2 for PostgreSQL or pymysql 
for MySQL. For SQLite this is included in Python’s standard library by default. You can find an overview of 
supported drivers for each SQL dialect in the SQLAlchemy docs.

The key functions are:
read_sql_table(table_name, con[, schema, ...])
Read SQL database table into a DataFrame.
read_sql_query(sql, con[, index_col, ...])
Read SQL query into a DataFrame.
read_sql(sql, con[, index_col, ...])
Read SQL query or database table into a DataFrame.
DataFrame.to_sql(name, con[, flavor, ...])
Write records stored in a DataFrame to a SQL database.
'''
engine = create_engine('sqlite:///:memory:')
df = pd.read_csv('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/example.csv')
print(df)
df.to_sql('data', engine)
sql_df = pd.read_sql('data', con=engine)
print(sql_df)

'''
Excel
Pandas can read and write excel files, keep in mind, this only imports data. Not formulas or images, 
having images or macros may cause this read_excel method to crash.
'''
df = pd.read_excel('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/Excel_Sample.xlsx',
              sheet_name='Sheet1')
df.to_excel('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/Excel_Sample1.xlsx',
            sheet_name='Sheet1')

'''
Some Queries using Pandas
'''

# Read the datafile
pop = pd.read_csv('/Users/suvosmac/Documents/CodeMagic/MLPY-PythonCrashCourse/Data/'
                  'population_by_county.csv')
pop.head()
print(pop.columns)

# How many states are represented in the dataframe
pop['State'].nunique()
# Get a list/array of all states
pop['State'].unique()

# What are the five most common county names in the US?
pop['County'].value_counts().head()

# Another alternate long approach
pop.groupby('County').count().sort_values(by='State', ascending=False).head()

# Five most populated counties
pop.sort_values(by='2010Census', ascending=False).head()

# Five most populated states
pop.groupby('State').sum().sort_values(by='2010Census', ascending=False).head()

# how many counties have 2010 population greater than 1 million
len(pop[pop['2010Census'] > 1000000])
# Alternate approach
sum(pop['2010Census'] > 1000000)

# How many counties dont have the name 'County' in their name?


def check_county(name):
    return "County" not in name


sum(pop['County'].apply(check_county))

# Alternate approach using lambda expression
sum(pop['County'].apply(lambda name: "County" not in name))

# Add a column that calculates the percent change between 2010 Census and 2017 Population Estimate
pop['PercentChange'] = 100 * (pop['2017PopEstimate']-pop['2010Census']) /\
                       pop['2010Census']
pop.head()


# What States have the highest estimated percent change between 2010 Census and
# the 2017 Population estimate

# STEP GROUP BY STATE, TAKE THE SUM OF THE POP COUNTY
states = pop.groupby(by='State').sum()
print(states)

# STEP - Recalculate the  percent changes
states['NewPercentChange'] = 100 * (states['2017PopEstimate']-states['2010Census']
                                   ) / states['2010Census']
states.sort_values('NewPercentChange', ascending=False).head()
