# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:48:12 2017
 
Logistic Regression using Python
We'll be trying to predict a classification- survival or deceased.
Let's begin our understanding of implementing Logistic Regression in Python for classification.
We'll use a "semi-cleaned" version of the titanic data set,
 
@author: Subhankar
 
LOGREG-TitanicDemo.py
"""
 
# load the libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# Read the titanic train csv file
train = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Udemy/titanic_train.csv")
# examine the dataset
train.head()
 
# Lets do some Exploratory Data Analysis

# use a heatmap to see missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# We see around 20% of the age data have missing values and
# way too many data is missing for Cabin values
 
# Lets check the distribution of the target variable (survived/non-survived)
sns.countplot(x='Survived',data=train)
 
# Lets study the survival with Sex
sns.countplot(x='Survived',hue = 'Sex',data=train)
# We see that mjority survivors are females and majority non-survivors are males
 
# Lets study the survival with Pclass
sns.countplot(x='Survived',hue = 'Pclass',data=train)
# We see that majority of the non survivors belonged to the class-3 and majority of survivors
# they belong to higher classes
 
# Study the distribution of the age variable - for now ignoring the missing values
sns.set_style('whitegrid')
sns.distplot(train['Age'].dropna(),kde=False,bins=30)

# Lets study the spouse-sibling variable/attribute
sns.countplot(x='SibSp', data=train)
# It is observed that most of the passengers were travelling alone and in second few were travelling with just
# one co-passenger, possibly spouse. Very few were travelling with children

# Lets explore the fair column
train['Fare'].hist(bins=30, figsize=(10,4)) 
# we see most of the passengers paid fares at the low end, which explains the large number of class-3 
# passengers

# Data Cleaning, treatment of missing values
# we have seen earlier that age variable is missing in many places...
# we will study the age of the passengers by class through a box plot
sns.boxplot(x='Pclass',y='Age',data=train)
# we see that average age of passenger in class 1 is max and in class 3 it is minimum

# we will now write a function to impute the missing values of age by the average in that
# respective class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
        
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)
# Now re-run the heatmap for missing values again
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# The cabin column has way too many missing values to be of any use..
# dropping the cabin column
train.drop('Cabin',axis=1,inplace=True)
# drop the other minimal missing values here and there
train.dropna(inplace=True)

# creation of dummy variables for categorical variables
# we would just use one, because the other will be perfectly correlated with other(s)
sex = pd.get_dummies(train['Sex'],drop_first=True) 
embark = pd.get_dummies(train['Embarked'],drop_first=True)

# we will now add these columns
train = pd.concat([train,sex,embark],axis=1)

# Now we will drop the columns we dont need
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# Now lets check the head
train.head()
# PassengerId column is also essentially a sequence number column and is not very useful
# for any machine learning predictions
train.drop('PassengerId',axis=1, inplace=True)

# Now we will do a train vs test split
X = train.drop('Survived',axis=1)
Y = train['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

# Now we will run the model
from sklearn.linear_model import LogisticRegression
# Create an instance of the model
logmodel = LogisticRegression()
# now we will fit the model
logmodel.fit(X_train,Y_train)
# Now lets run from predictions
predictions = logmodel.predict(X_test)

# Now to run the diagnostics
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Alternate Approach for Running LogisticRegression
'''
import statsmodels.api as sm
X1 = train.drop('Survived',axis=1)
Y1 = train['Survived']
from sklearn.cross_validation import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 101)

# Now we will run the logit model
logit = sm.Logit(Y1_train,X1_train)
# fit the model
result = logit.fit()
# Lets view the results
print(result.summary())
# based on the result summary, we see that only Fare and Male column is statistically significant
# in making the predictions, dropping all other columns and running the model and to a near extent
# SibSp
X1 = train[['Fare','male','SibSp']]
# re-run the model
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 101)
logit = sm.Logit(Y1_train,X1_train)
result = logit.fit()
print(result.summary())
# Confidence interval gives you an idea for how robust the coefficients of the model are
print (result.conf_int())
'''
For example we see that there is an inverse relationship between being a male and probability
of Survival. The chances of Survival goes down if the passenger is a male.
There is a positive relationship between Fare and Survival - Higher fare, higher chances of survival
There is an inverse relationship between having Siblings/Spouse and Survival ...
More number of Siblings and Spouse, lesser is the chance of passenger survival
'''
# Lets inspect the odds ratio which will tell us how a 1 unit 
# increase or decrease in a variable affects the odds of Survival.
print(np.exp(result.params))

# Now lets do the predictions and concat the result with the original dataset
survival_prob = result.predict(X1_test)
X1_test = pd.concat([X1_test,survival_prob,Y1_test],axis=1)
X1_test.rename(columns={0:'Survival Prob'},inplace=True)
# Now X1 has the complete predictions as well as actuals...

# Lets run some visualizations and see the probabilities
sns.pointplot(x='Fare',y='Survival Prob',hue='male',data=X1_test)
sns.violinplot(x='male',y='Survival Prob',data=X1_test)
sns.violinplot(x='SibSp',y='Survival Prob',data=X1_test)
