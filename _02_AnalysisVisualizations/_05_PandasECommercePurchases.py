"""
Demonstrates How to work with Pandas on the ECommercePurchases dataset
Some data analysis on the dataset

@author: suvosmac
"""

import numpy as np
import pandas as pd

# Read the dataset
ecom = pd.read_csv('/Volumes/Data/CodeMagic/Data Files/ECommercePurchases')

# Check the head of the data frame being created
ecom.head()
# Check the number of rows and columns
ecom.info()

# how to check only the number of columns
len(ecom.columns)
# Check only the number of rows
len(ecom.index)

# What is the average purchase price
print(ecom['Purchase Price'].mean())
# What is the highest and lowest purchase price
print(ecom['Purchase Price'].max())
print(ecom['Purchase Price'].min())

# How many people have English 'en' as their language choice on the website
print(ecom[ecom['Language'] == 'en']['Language'].count())

# How many people have job title as "Lawyer"
ecom[ecom['Job'] == 'Lawyer']['Job'].count()

# How many people made the purchase during the AM and how many during the PM
ecom['AM or PM'].value_counts()

# What are the top five most common job titles
ecom['Job'].value_counts().head(5)

'''
Someone made a purchase that came from Lot:"90 WT", what was the purchase price for this
transaction
'''
print(ecom[ecom['Lot'] == '90 WT']['Purchase Price'])

# What is the email of the person with the following credit card number - 492653524267853
print(ecom[ecom['Credit Card'] == 4926535242672853]["Email"])

'''
How many people have American Express as their credit card and have made
purchases above $95
'''

len(ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95.0)].index)

# How many people have credit card that expires on 2025
# We have to apply some string formatting on the exp data column
len(ecom[ecom['CC Exp Date'].apply(lambda exp: exp[3:]) == '25'])

# What are the top 5 most popular email providers/hosts
ecom['Email'].apply(lambda exp: exp.split('@')[1]).value_counts().head(5)
