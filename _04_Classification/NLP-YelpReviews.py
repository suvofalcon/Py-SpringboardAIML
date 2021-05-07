#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 21:13:32 2018

In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off 
the text content in the reviews. This will be a simpler procedure than the lecture, since we will utilize the 
pipeline methods for more complex tasks.

We will use the Yelp Review Data Set from Kaggle.

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. 
(Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users.
All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. 
In other words, it is a rating of the review itself, not a rating of the business.
The "useful" and "funny" columns are similar to the "cool" column.

@author: suvosmac
"""
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# read the data
yelp = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/yelp.csv")

# Check the data load
yelp.head()
yelp.info()

# Summary statistics
yelp.describe()

# Create a new column called "text length' which is the number of words in the text column
yelp['text length'] = yelp['text'].apply(len)

'''
Exploratory Data Analysis
'''
# Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. 
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')

# Create a boxplot of text length for each star category.
sns.boxplot(x='stars',y='text length',data=yelp)

# Count plot by stars rating - more number of 4 and 5 star reviews
sns.countplot(x='stars',data=yelp)

# we will use groupby to get the mean values of the numerical column
stars = yelp.groupby('stars').mean()
stars

# use correlation to see
stars.corr()
# we will use a heatmap to for the correlation
sns.heatmap(stars.corr(),annot=True,cmap='coolwarm')

# Now we will create a function which can be applied to the entire dataframe
from nltk.corpus import stopwords
def text_process(mess):
    """
    1. remove punc
    2. remove stopwords
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    return clean_words

'''
NLP Classification Task
'''

# To make things easier, we will go ahead and grab reviews which are only 1 star and 5 stars
# we will subset another dataframe
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

# Check the head
yelp_class.head()

# Seggregate the feature and response variable
X = yelp_class['text']
y = yelp_class['stars']

# Import CountVectorizer and create a CountVectorizer object.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). 
X = cv.fit_transform(X)

'''
Train, test split
Training the model
Predictions and Evaluations
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

# Training the model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# fit the training data
nb.fit(X_train,y_train)

# Now we will run predictions
predictions = nb.predict(X_test)

# Create a classification report and confusionMatrix
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

'''
Let's see what happens if we try to include TF-IDF to this process using a pipeline
'''
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# Create the pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

'''
Train Test split
'''
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

# Fit the pipeline to the training data
pipeline.fit(X_train,y_train)

'''
Predictions and Evaluations
'''
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))