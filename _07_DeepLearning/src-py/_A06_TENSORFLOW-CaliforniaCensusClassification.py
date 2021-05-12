"""
We will demonstrate tensor flow for a Classification

We'll be working with some California Census Data, we'll be trying to use various features of an individual to
predict what class of income they belogn in (>50k or <=50k).

Data information as below

Column Name	               Type	                            Description
age	                       Continuous	                    The age of the individual
workclass	               Categorical	                    The type of employer the individual has
                                                            (government, military, private, etc.).
fnlwgt	                   Continuous	                    The number of people the census takers believe that
                                                            observation represents (sample weight).
                                                            This variable will not be used.
education	               Categorical	                    The highest level of education achieved for that individual.
education_num	           Continuous	                    The highest level of education in numerical form.
marital_status	           Categorical	                    Marital status of the individual.
occupation	               Categorical	                    The occupation of the individual.
relationship	           Categorical	                    Wife, Own-child, Husband, Not-in-family, Other-relative,
                                                            Unmarried.
race	                   Categorical	                    White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
gender	                   Categorical	                    Female, Male.
capital_gain	           Continuous	                    Capital gains recorded.
capital_loss	           Continuous	                    Capital Losses recorded.
hours_per_week	           Continuous	                    Hours worked per week.
native_country	           Categorical	                    Country of origin of the individual.
income	                   Categorical	                    ">50K" or "<=50K", meaning whether the
                                                            person makes more than $50,000 annually.

"""

import pandas as pd
import tensorflow as tf


# read the data
census = pd.read_csv("/Users/suvosmac/Documents/CodeMagic/DataFiles/Udemy/census_data.csv")

# Check the data load

census.head()

# We will have to recode the target variable 'income' into 0 and 1. Tensorflow will not understand string values
census['income_bracket'].unique()

census['income_bracket'] = census['income_bracket'].apply(lambda label: int(label ==' >50K'))

'''
Perform the train, test split on the data
'''
from sklearn.model_selection import train_test_split
x_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)

'''
Feature Engineering 
Create the feature columns
'''

# Check on the columns
census.columns
census.info()

# Build the feature columns
age = tf.feature_column.numeric_column("age")
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=100)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=100)
education_num = tf.feature_column.numeric_column("education_num")
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status",hash_bucket_size=100)
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=100)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=100)
race = tf.feature_column.categorical_column_with_hash_bucket("race",hash_bucket_size=100)
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",["Male","Female"])
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country",hash_bucket_size=100)

# build the list
feat_cols = [age,workclass,education,education_num,marital_status,occupation,relationship,race,gender,
             capital_gain,capital_loss,hours_per_week,native_country]

"""
Build the model

- create the input function
- train the model
- evaluate the model
"""

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=1000,
                                                 shuffle=True)
# we will use a linear Classifier to begin with
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

# train the model
model.train(input_fn=input_func,steps=10000)

# build the prediction function
pred_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=100,num_epochs=1,shuffle=False)

# do the predictions
predictions = list(model.predict(input_fn=pred_func))

# see the predictions
print(predictions)

# extract the predicted values
final_pred = []
for pred in predictions:
    final_pred.append(pred['class_ids'][0])

# lets check the classification report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,final_pred))
print("\n")
print("Confusion Matrix")
print(confusion_matrix(y_test,final_pred))

'''
Re-do the whole modelling with a densely connected neural network
'''
# Build the feature columns
# for categorical columns in a densely connected neural networks - it has to be embedded with an embedding_column
# or indicator_column

age = tf.feature_column.numeric_column("age")

workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=100)
embedded_workclass = tf.feature_column.embedding_column(categorical_column=workclass,dimension=100)

education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=100)
embedded_education = tf.feature_column.embedding_column(categorical_column=education,dimension=100)

education_num = tf.feature_column.numeric_column("education_num")

marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status",hash_bucket_size=100)
embedded_marital_status = tf.feature_column.embedding_column(categorical_column=marital_status,dimension=100)

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=100)
embedded_occupation = tf.feature_column.embedding_column(categorical_column=occupation,dimension=100)

relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=100)
embedded_relationship = tf.feature_column.embedding_column(categorical_column=relationship,dimension=100)

race = tf.feature_column.categorical_column_with_hash_bucket("race",hash_bucket_size=100)
embedded_race = tf.feature_column.embedding_column(categorical_column=race,dimension=100)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",["Male","Female"])
indicator_gender = tf.feature_column.indicator_column(gender)

capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country",hash_bucket_size=100)
embedded_native_country = tf.feature_column.embedding_column(categorical_column=native_country,dimension=100)

# build the list
feat_cols = [age,embedded_workclass,embedded_education,education_num,embedded_marital_status,embedded_occupation,embedded_relationship
            ,embedded_race,indicator_gender,capital_gain,capital_loss,hours_per_week,embedded_native_country]

# Build the input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=1000,
                                                 shuffle=True)

# we will build a densely connected neural networks with a three layers and each having 6 neurons
dnn_model = tf.estimator.DNNClassifier(hidden_units=[6,6,6],feature_columns=feat_cols,
                                       n_classes=2)

dnn_model.train(input_fn=input_func,steps=20000)

# build the prediction function
pred_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=100,num_epochs=1,shuffle=False)

# do the predictions
predictions = list(model.predict(input_fn=pred_func))

# extract the predicted values
final_pred = []
for pred in predictions:
    final_pred.append(pred['class_ids'][0])

# lets check the classification report
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,final_pred))
print("\n")
print("Confusion Matrix")
print(confusion_matrix(y_test,final_pred))


'''
Realtiy check with a random forest classifier
'''

# read the data
census = pd.read_csv("/Users/suvosmac/Documents/CodeMagic/DataFiles/Udemy/census_data.csv")

census.columns
census['workclass'] = census['workclass'].astype('category')
census['education'] = census['education'].astype('category')
census['marital_status'] = census['marital_status'].astype('category')
census['occupation'] = census['occupation'].astype('category')
census['relationship'] = census['relationship'].astype('category')
census['race'] = census['race'].astype('category')
census['gender'] = census['gender'].astype('category')
census['native_country'] = census['native_country'].astype('category')

census['income_bracket'] = census['income_bracket'].apply(lambda label: int(label ==' >50K'))

# create dummy variables for categorical columns (since the labels are non-numeric)
final_data = pd.get_dummies(census,columns=['workclass','education','marital_status','occupation','relationship','race','gender',
                                            'native_country'],drop_first=True)

from sklearn.model_selection import train_test_split
x_data = final_data.drop('income_bracket',axis=1)
y_labels = final_data['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
# Fit the classifier to training data
rf.fit(X_train,y_train)
# Run the predictions
rf_preds = rf.predict(X_test)
# Classification report and confusion matrix
print(classification_report(y_test,rf_preds))
print(confusion_matrix(y_test,rf_preds))