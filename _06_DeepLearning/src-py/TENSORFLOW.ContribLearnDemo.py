#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 20:13:38 2018

As we saw previously how to build a full Multi-Layer Perceptron model with full 
Sessions in Tensorflow. Unfortunately this was an extremely involved process. 
However developers have created ContribLearn (previously known as TKFlow or SciKit-Flow) which 
provides a SciKit Learn like interface for Tensorflow!

It is much easier to use, but you sacrifice some level of customization of your model.

@author: suvosmac
"""
import tensorflow as tf
import pandas as pd

# Load the famous iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# check the keys, since this is a dictionary
iris.keys()

# now to see the target variables
iris['target']
iris['target_names'] 

# Now create a data frame from this dictionary
irisdf_features = pd.DataFrame(iris['data'],columns=iris['feature_names'])
irisdf_target = pd.DataFrame(iris['target'],columns=['Species'])
#irisdf_target = irisdf_target.replace({'Species':{0 : 'setosa', 1: 'versicolor', 2: 'virginica'}}, inplace=True)
irisdf = pd.concat([irisdf_features,irisdf_target],axis = 1)
# check the initial rows
irisdf.head()
irisdf.info()

# We will grab the data and target
X = irisdf.drop('Species',axis=1)
y = irisdf['Species']

# We will do a train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)

'''
We will build a tensor flow model, just like a normal scikit learn model
'''
import tensorflow.contrib.learn as learn
'''
we will use DNNClassifier, which stands for Deep Neural Network:
hidden_units = How many nodes per layer
'''
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

classifier = learn.DNNClassifier(hidden_units=[20,30,40],n_classes = 3,feature_columns=feature_columns)

classifier.fit(X_train, y_train, steps=300, batch_size=50)

# Since it returns a generator object, converting it to list, so as to use in classfication report and confusion matrix
iris_predictions = list(classifier.predict(X_test))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,iris_predictions))
print(confusion_matrix(y_test,iris_predictions))
