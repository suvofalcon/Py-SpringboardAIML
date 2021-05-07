# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:59:33 2017
SVM-BreastCancer.py

Demonstration of Classification using Support Vector Machines on a
Breast Cancer dataset

@author: Subhankar
"""

# Import of libraries
import pandas as pd

# Load the dataset - breast cancer dataset from scikit library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# This is a dictionary ... to view the keys
cancer.keys()
# to see the detailed description of the data
print(cancer['DESCR'])

# Now create a data frame from this dictionary
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
# check the initial rows
df_feat.head()
df_feat.info()

# now to see the target variables
cancer['target']
cancer['target_names'] # 0 is malignant and 1 is benign

# Now we will create the target data frame
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
df_target.info()

# Train-test split
X = df_feat
y = cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)

# Now we will train the support vector classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

# Predictions and Evaluations
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# The default model fared very badly and it means the model needs to have it parameters 
# adjusted (it may also help to normalize the data). We can search for parameters using a GridSearch!

'''
Among many parameters for running rhe support vector model, few key ones are as below
c - controls the cost of misclassfication on the training data. A large c value gives low bias and high variance, because we
    penalise the cost of misclassification. similarly a low c value gives higher bias and low variance
    
gamma - high value results in the high bias and low variance, which means the support vector doesnt have a widespread influence
'''

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=100)
grid.fit(X_train,y_train)

# to get the best parameter
grid.best_params_

# Now we will re-run based on the best parameter combination
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))













