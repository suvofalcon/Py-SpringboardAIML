#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 09:33:43 2018
 
We will use the Bank Authentication Data Set
The data consists of 5 columns
- variance of wavelet Transformed image (continuous)
- Skewness of wavelet Transformed image (continuous)
- curtosis of wavelet Transformed image (continuous)
- entropy of image (continous)
- class (integer)

Where class indicates whether or not the Bank Note was authentic

@author: suvosmac
"""

# import libraries to be used
import pandas as pd
import seaborn as sns
import tensorflow as tf
'''
Load the data and some exploratory analysis
'''

# get the data
data = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Udemy/bank_note_data.csv")
# Check the head of the data
data.head()

# Check the countplot of the classes
sns.countplot(x='Class',data=data)

# create a pairplot of the data with hue as class
sns.pairplot(data,hue='Class')

'''
Data Preparation
'''
# we will scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
# create a dataframe out of this scaled features
df_features = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_features.head()

'''
Do a train, test split
'''
X = df_features
y = data['Class']

# Tensor flow works best when data is input as Numpy array instead of pandas series
X = X.values
y = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42)

'''
Build the tensor flow model
'''
import tensorflow.contrib.learn as learn
# Build the classifier
feature_columns = learn.infer_real_valued_columns_from_input(X_train)
classifier = learn.DNNClassifier(hidden_units=[10,20,30],n_classes = 2,
                                 feature_columns=feature_columns,
                                 activation_fn=tf.nn.relu)

# Fit the classifier to the training data
import datetime
start = datetime.datetime.now()
print("The Model training started at :",start)
classifier.fit(X_train, y_train, steps=200, batch_size=20)
end = datetime.datetime.now()
print("The time taken for training the model is :",(end - start))

'''
Model evaluation
'''
# Since it returns a generator object, converting it to list, so as to use in classfication report and confusion matrix
class_predictions = list(classifier.predict(X_test))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,class_predictions))
print(confusion_matrix(y_test,class_predictions))

'''
Realtiy check with a random forest classifier
'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
# Fit the classifier to training data
rf.fit(X_train,y_train)
# Run the predictions
rf_preds = rf.predict(X_test)
# Classification report and confusion matrix
print(classification_report(y_test,rf_preds))
print(confusion_matrix(y_test,rf_preds))