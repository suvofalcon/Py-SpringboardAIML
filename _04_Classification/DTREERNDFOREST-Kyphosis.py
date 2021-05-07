# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:48:53 2017
DTREERNDFOREST-Kyphosis.py

Demonstration of Decision Trees and Random Forests in Python

@author: Subhankar
"""

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the data
df = pd.read_csv('//Volumes/Data/CodeMagic/Data Files/Udemy/kyphosis.csv')

# check the data
df.head()

# We will do some EDA on the data and run a pairplot
sns.pairplot(df,hue='Kyphosis')
# lets understand the distribution of the target variable
sns.countplot(df['Kyphosis'])

# Train Test Split
# Let's split up the data into a training set and a test set!

from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Decision Trees¶
# We'll start just by training a single decision tree.
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Prediction and Evaluation
# Let's evaluate our decision tree.
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix     
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Random Forests¶
# Now let's compare the decision tree model to a random forest.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
