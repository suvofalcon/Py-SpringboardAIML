# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:27:11 2017
DTREERNDFOREST-LendingData.py
 
For this project we will be exploring publicly available data from LendingClub.com.
Lending Club connects people who need money (borrowers) with people who have money (investors).
Hopefully, as an investor you would want to invest in people who showed a profile of having a high
probability of paying you back. We will try to create a model that will help predict this.
Lending club had a very interesting year in 2016, so let's check out some of their data and keep the context in mind.
This data is from before they even went public.
We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower
paid back their loan in full. You can download the data from here or just use the csv already provided.
It's recommended you use the csv provided as it has been cleaned of NA values.
 
Here are what the columns represent:
credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
 
purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational",
         "major_purchase", "small_business", and "all_other").
 
int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11).
        Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
       
installment: The monthly installments owed by the borrower if the loan is funded.
 
log.annual.inc: The natural log of the self-reported annual income of the borrower.
 
dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
 
fico: The FICO credit score of the borrower.
 
days.with.cr.line: The number of days the borrower has had a credit line.
 
revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
 
revol.util: The borrower's revolving line utilization rate (the amount of the credit line used
        relative to total credit available).
 
inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
 
delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
 
pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
 
@author: Subhankar
"""
 
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# lets load the data
loans = pd.read_csv('D:\\CodeMagic\\loan_data.csv')
 
# lets check the data load
loans.head()
loans.info()
# run the summary statistics
loans.describe()
 
# Exploratory Data Analysis
 
# Distribution of our target variable
sns.countplot(loans['not.fully.paid']) # highly unbalanced data
# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
loans[loans['credit.policy']==1]['fico'].hist(bins=30,color='blue',label='Credit Policy = 1',
     alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=30,color='red',label='Credit Policy = 0',
     alpha=0.6)
plt.legend()
plt.xlabel('FICO')
# It is quite evident from the visualization above, that we have more people in the data set who has a
# credit policy than the otherwise..
# Also any individual having a FICO score of 650 or less will not be eligible for credit policy
 
# Create a similar figure, except this time select by the not.fully.paid column.
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,color='blue',label='Not Fully Paid = 1',
     alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,color='red',label='Not fully Paid = 0',
     alpha=0.6)
plt.legend()
plt.xlabel('FICO')
 
# Create a countplot using seaborn showing the counts of loans by purpose,
# with the color hue defined by not.fully.paid.
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue = 'not.fully.paid',data = loans,palette='Set1')
 
# Let's see the trend between FICO score and interest rate.
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
# above shows a clear trend that as the fico score increases the interest rate comes down
 
# Create lmplots to see if the trend differed between not.fully.paid and credit.policy.
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
 
# Setting up the Data
# Let's get ready to set up our data for our Random Forest Classification Model!
# Since we have a categorical column, 'Purpose' we would have to build dummy variables
cat_feats = ['purpose']
# Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe
# that has new feature columns with dummy variables. Set this dataframe as final_data.
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
# Check the final data
final_data.info()
 
# Train Test Split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
 
# Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
 
# Predictions and Evaluation of Decision Tree
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
 
# Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
# Predictions and Evaluation
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))