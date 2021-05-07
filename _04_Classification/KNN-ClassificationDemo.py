# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:58:30 2017
@author: Subhankar
 
This code demonstrates the Python implementation for classification using KNN algorithm
 
"""
 
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# lets assume that the dataset we have is anonymous and lets load the data
# lets prevent treating the index column as a separate column
df = pd.read_csv("//Volumes/Data/CodeMagic/Data Files/Udemy/Classified Data", index_col=0)
# check the data initial rows
df.head()
 
# Since KNN is a geometrical algorithm, we would first proceed with scaling the data
from sklearn.preprocessing import StandardScaler
# create an instance of StandardScaler
scaler = StandardScaler()
# to scale
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
 
# now create the dataframe with the scaled values
df_features = pd.DataFrame(scaled_features, columns = df.columns[:-1])
# now check the initial rows of the scaled feature data frame
df_features.head()
 
# Now train, test split
from sklearn.model_selection import train_test_split
features = df_features
target = df['TARGET CLASS']
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3,
                                                                            random_state = 101)
 
# KNN implementation
from sklearn.neighbors import KNeighborsClassifier
# instantiate the model and we start with K = 1
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(features_train, target_train)
 
# Now do the predictions, post which we will do evaluation
pred = knn.predict(features_test)
 
# For evaluation of the model, we will use the elbow method to chose the correct K value
from sklearn.metrics import classification_report, confusion_matrix
# lets print the confusion matrix and classification report
print(confusion_matrix(target_test,pred))
print(classification_report(target_test,pred))
 
# The above model seems to be good with k=1 already... but we will use the elbow method
# to chose the correct k values. We will use many k values and run the model and plot
# their error rate

# create an empty dataframe to have the kvalues and error_rates
errordf = pd.DataFrame(columns=['kvalue','error_rate'])

for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features_train,target_train)
    pred_k = knn.predict(features_test)
    errordf.loc[len(errordf)] = [k,np.mean(pred_k)]
 
# Now we will plot the error rate
plt.plot(errordf['kvalue'],errordf['error_rate'],linestyle='dashed',marker='o',
         markerfacecolor='red',markersize=8)
plt.title('Error Rate vs K-Value')
plt.xlabel('K-Values')
plt.ylabel('Error Rate')

# Looking at the graph, we will chose the K value at min error_rate here
 
# Re-run the modelnwith k-value which haw minimum error rate
# we extract out from data frame, the finalK would be a series, so we need to extract the item
finalK = errordf[errordf['error_rate'] == errordf['error_rate'].min()]['kvalue'].item()
knn = KNeighborsClassifier(n_neighbors = int(finalK))
knn.fit(features_train, target_train)
pred = knn.predict(features_test)
print(confusion_matrix(target_test,pred))
print('\n')
print(classification_report(target_test,pred)) # accuracy is now 95%