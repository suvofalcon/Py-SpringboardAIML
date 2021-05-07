# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:19:39 2017
SVM-Iris.py

Demonstration of Classification using Support Vector Machines on Iris dataset

@author: Subhankar
"""

# Import of libraries
import pandas as pd
import seaborn as sns

# Get the data
# iris = sns.load_dataset('iris') # since this is not working in cog machine
from sklearn import datasets
iris = datasets.load_iris()
# This is a dictionary ... to view the keys
iris.keys()
# to see the detailed description of the data
print(iris['DESCR'])

# now to see the target variables
iris['target']
iris['target_names'] 

# Now create a data frame from this dictionary
irisdf_features = pd.DataFrame(iris['data'],columns=iris['feature_names'])
irisdf_target = pd.DataFrame(iris['target'],columns=['Species'])
irisdf_target.replace({'Species':{0 : 'setosa', 1: 'versicolor', 2: 'virginica'}}, inplace=True)
irisdf = pd.concat([irisdf_features,irisdf_target],axis = 1)
# check the initial rows
irisdf.head()
irisdf.info()

# Exploratory Data Analysis
# Create a pairplot of the data set. Which
sns.pairplot(irisdf, hue='Species', palette='Dark2') # Setosa is most separable
# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = irisdf[irisdf['Species']=='setosa']
sns.kdeplot( setosa['sepal width (cm)'], setosa['sepal length (cm)'],
                 cmap="plasma", shade=True, shade_lowest=False)

# Train Test Split
from sklearn.model_selection import train_test_split
X = irisdf.drop('Species',axis=1)
y = irisdf['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)

# Train a Model
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

# Model Evaluation
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))