#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:59:03 2018

In this project we will be working with a fake advertising data set, indicating whether or 
not a particular internet user clicked on an Advertisement on a company website. We will try 
to create a model that will predict whether or not they will click on an ad based off the 
features of that user.

This data set contains the following features:
    
'Daily Time Spent on Site': consumer time on site in minutes
'Age': cutomer age in years
'Area Income': Avg. Income of geographical area of consumer
'Daily Internet Usage': Avg. minutes a day consumer is on the internet
'Ad Topic Line': Headline of the advertisement
'City': City of consumer
'Male': Whether or not consumer was male
'Country': Country of consumer
'Timestamp': Time at which consumer clicked on Ad or closed window
'Clicked on Ad': 0 or 1 indicated clicking on Ad

@author: suvosmac
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
advertising = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Udemy/advertising.csv")
# Lets check the head of the data
advertising.head()
# quickly check the summary
advertising.info()

'''
Exploratory Data Analysis
'''
# Explore the response variable
sns.countplot(advertising['Clicked on Ad'])
print(advertising['Clicked on Ad'].value_counts())
# We see there is a equal proportion of people who have clicked on ads and who did not

# Explore the age variable
sns.distplot(advertising['Age'])

# Create a joint plot between Age and Area income
sns.jointplot(x='Age',y='Area Income',data=advertising)

# Create a joint plot showing the kde distributions of Daily Time spent on site vs. Age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=advertising,kind='kde',
              color = 'red')

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',
              data=advertising,color='green')

# create a pairplot with the hue defined by the 'Clicked on Ad' column feature
sns.pairplot(data=advertising,hue='Clicked on Ad')
sns.pairplot(data=advertising,x_vars='Age',y_vars='Area Income',hue='Clicked on Ad')

'''
Logistic Regression
'''

# Do a train, test split
from sklearn.model_selection import train_test_split

X = advertising[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']]
y = advertising[['Clicked on Ad']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
# Create an instance of the model
logmodel = LogisticRegression()
# now we will fit the model
logmodel.fit(X_train,y_train)
# Now lets run from predictions
predictions = logmodel.predict(X_test)

'''
Predictions and Evaluations
'''
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

'''
Alternate approach of logistic modelling and evaluation
'''
import statsmodels.api as sm
# Now we will run the logit model
logit = sm.Logit(y_train,X_train)
# fit the model
result = logit.fit()
# Lets view the results
print(result.summary())
# From the summary we see that, whether or not the consumer is male, is not 
# statistically significant in predicting the click, So dropping that variable
# and re-run the model

X = advertising[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage']]
y = advertising[['Clicked on Ad']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=42)
logit = sm.Logit(y_train,X_train)
result = logit.fit()
print(result.summary())
# Confidence interval gives you an idea for how robust the coefficients of the model are
print (result.conf_int())

# Now lets do the predictions and concat the result with the original dataset
click_prob = result.predict(X_test)
X_test = pd.concat([X_test,click_prob,y_test],axis=1)
X_test.rename(columns={0:'Click Prob'},inplace=True)

# Visualize daily time spent on site and predicted probabilities
sns.pointplot(x='Daily Time Spent on Site',y='Click Prob',data=X_test)
# We see as the daily time spent on site increases, the probability decreases