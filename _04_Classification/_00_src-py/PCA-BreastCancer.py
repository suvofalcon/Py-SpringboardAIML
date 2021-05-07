# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:35:35 2018

Demonstration of Principal Component Analysis on Breast Cancer Dataset
PCA is an unsupervised algorithm

@author: 132362
"""

# Import needed libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset - breast cancer dataset from scikit library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# This is a dictionary ... to view the keys
cancer.keys()
# to see the detailed description of the data
print(cancer['DESCR'])

# Now create a data frame from this dictionary
df= pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
# check the initial rows
df.head()
df.info()

# Before performing PCA it is important to standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

'''
Perform PCA
'''
from sklearn.decomposition import PCA
# we are saying we want to keep only 2 principal components and visualize the entire
# 30 dimensions into 2 principal components
pca = PCA(n_components=2) 
pca.fit(scaled_data)

# Now we will transform this fitted data into principal components
x_pca = pca.transform(scaled_data)
# Check the scaled data shape
scaled_data.shape
# Check the shape after the transformation
x_pca.shape

# Now we will plot out these dimensions
plt.figure(figsize=(8,6))
#x_pca is a numpy array of two dimensions, so we want to plot all the rows for column 0 vs
# all the rows for column 1
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

# This technique is so powerful, that we see just with the help of just two principal components,
# we have a very clear separation between two target classes

# To see these components
pca.components_
# This is a numpy matrix, where each row represents the principal component and each column actually relates 
# to the actual features

# Lets create a dataframe to understand this
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
df_comp
# Now we will try to visualize this using a heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')





