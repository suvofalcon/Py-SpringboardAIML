"""
Demonstrates How to work with Pandas on the SF Salaries dataset
Some data analysis on the dataset

@author: suvosmac
"""

import pandas as pd

# Read the SFSalaries dataset

sal = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/SFSalaries.csv')
sal.head()

"""
Use the info method to find out how many entries are there.
In short this will tell us a brief summary about the data frame

"""
sal.info()

# What is the average Base pay
sal['BasePay'].mean()

# What is the highest amount of Overtime Pay in the dataset
sal['OvertimePay'].max()

# What is the Job title of JOSEPH DRISCOLL
# All details of Joseph Driscoll
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL'])
# Just the job title
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle'])

# How much does Joseph Driscoll make
print(sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits'])

# What is the name of the highest paid person including benefits
print(sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName'])
# an alternate approach
print(sal.loc[sal['TotalPayBenefits'].idxmax()]['EmployeeName'])


# What is the name of the lowest paid person (including benefits)?
# Conventional approach
print(sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].min()]['EmployeeName'])
print(sal.loc[sal['TotalPayBenefits'].idxmin()]['EmployeeName'])

# What was the average (BasePay) of all employees per year
print(sal.groupby('Year').mean()['BasePay'])

# How many unique job titles are there
print(sal['JobTitle'].nunique())

# What are the top 5 most common jobs
# value_counts() will arrange it in descending order
print(sal['JobTitle'].value_counts().head(5))

"""
How many job titles were represented by only one person in 2013, which means
job titles with only one occurence
"""

print(sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1))

# How many people have Chief in their job title


def chief_string(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False


# Now we will apply this function
print(sum(sal['JobTitle'].apply(lambda x:chief_string(x))))

# Is there a correlation between length of the Job Title and Salary
# We will create a column based on the length of the job title
sal['title_len'] = sal['JobTitle'].apply(len)

# Now we will display the job title and title length
sal[['JobTitle', 'title_len']]
# Now to get the correlation
sal[['title_len', 'TotalPayBenefits']].corr()  # we see no correlation
