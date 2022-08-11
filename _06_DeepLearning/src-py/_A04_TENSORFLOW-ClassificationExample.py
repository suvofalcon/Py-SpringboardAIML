# -*- coding: utf-8 -*-
"""
We will demonstrate tensor flow for a classification example

We will use the TF Estimator API
We will deal with Categorical and Numerical features
Linear Classifier and DNNClassifier

Title: Pima Indians Diabetes Database

Several constraints were placed on the selection of these instances from a larger database. In particular, all patients
here are females at least 21 years old of Pima Indian heritage.

Number of Instances: 768
Number of Attributes: 8 plus class
For Each Attribute: (all numeric-valued)
Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
Age (years)
Class variable (0 or 1)

Class Distribution: (class value 1 is interpreted as "tested positive for diabetes")

@author: suvosmac
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Add the data
diabetes = pd.read_csv("/Users/suvosmac/Documents/CodeMagic/DataFiles/Udemy/pima-indians-diabetes.csv")

# Check the data load
diabetes.head()
diabetes.info()

# Check the column names
print(diabetes.columns)

# We will extract out only the columns which we will normalize
# (essentially the numerical columns and not categorical columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

# We will normalize/standardize the columns using pandas
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

# Now lets inspect the data again
diabetes.head()

# Build the feature columns for Tensorflow modelling
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Feature columns building for categorical attributes
'''
If you know the set of all possible feature values of a column and there are only a few of them, you can use 
categorical_column_with_vocabulary_list.
If you don't know the set of possible values in advance you can use categorical_column_with_hash_bucket
'''
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# Alternative - size could be any number but greater than the actual number of unique values in the categorical column
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

'''
Converting a Continuous column to a Categorical Column for ex-Age
'''
# Lets study the age variable
diabetes['Age'].hist(bins=20)
plt.show()

# I will create a bucketized column for the variable age
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

'''
Consolidate all feature columns
'''
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,age_bucket,assigned_group]

'''
Perform a train, test split
'''
x_data = diabetes.drop('Class',axis=1)
labels = diabetes['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.3, random_state=101)

'''
Perform the tensorflow modelling with a simple Linear Classifier
'''

# Create the input function to be used in tensorflow estimator
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

# create the model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)

# train the model
model.train(input_fn=input_func,steps=1000)

# evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
results = model.evaluate(eval_input_func)

# see the results
print(results)

# Get predictions from this model - There is no y value here, as that would be predicted
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)

predictions = list(model.predict(pred_input_func))
print(predictions) # it shows the predictions and also the associated probabilities for the predictions

'''
Now we will do tensor flow modelling with a dense neural network classifier
'''

# for categorical columns in a densely connected neural networks - it has to be embedded with an embedding_column
# or indicator_column
embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,age_bucket,embedded_group_col]

# Create the input function to be used in tensorflow estimator
input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)

# we will build a densely connected neural networks with a six and each having 20 neurons
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10,10,10,10],feature_columns=feat_cols,
                                       n_classes=2)
# train the model
dnn_model.train(input_fn=input_func,steps=1000)

# evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(X_test,y_test,batch_size=10,num_epochs=1,shuffle=False)
results = dnn_model.evaluate(eval_input_func)

# see the results
print(results)