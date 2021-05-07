# -*- coding: utf-8 -*-
"""
For this project we will attempt to use KMeans Clustering to cluster Universities into to 
two groups, Private and Public.
It is very important to note, we actually have the labels for this data set, 
but we will NOT use them for the KMeans clustering algorithm, since that is an 
unsupervised learning algorithm.

When using the Kmeans algorithm under normal circumstances, it is because you 
don't have labels. In this case we will use the labels to try to get an idea of 
how well the algorithm performed, but you won't usually do this for Kmeans, 
so the classification report and confusion matrix at the end of this project, 
don't truly make sense in a real world setting!.

The Data
We will use a data frame with 777 observations on the following 18 variables.

Private A factor with levels No and Yes indicating private or public university
Apps Number of applications received
Accept Number of applications accepted
Enroll Number of new students enrolled
Top10perc Pct. new students from top 10% of H.S. class
Top25perc Pct. new students from top 25% of H.S. class
F.Undergrad Number of fulltime undergraduates
P.Undergrad Number of parttime undergraduates
Outstate Out-of-state tuition
Room.Board Room and board costs
Books Estimated book costs
Personal Estimated personal spending
PhD Pct. of faculty with Ph.D.’s
Terminal Pct. of faculty with terminal degree
S.F.Ratio Student/faculty ratio
perc.alumni Pct. alumni who donate
Expend Instructional expenditure per student
Grad.Rate Graduation rate

@author: suvosmac
"""

# import the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data file
df = pd.read_csv("/Volumes/Data/CodeMagic/Data Files/Udemy/College_Data",index_col=0)

# Check the head and some basic statistics on the data
df.head()
df.info()
df.describe()

'''
Perform some EDA
'''
# Create a scatterplot of Grad.Rate versus Room.Board where 
# the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df,hue='Private',palette='coolwarm',
            fit_reg=False)
plt.show()

# Create a scatterplot of F.Undergrad versus Outstate 
# where the points are colored by the Private column.
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df,hue='Private',palette='coolwarm',
            fit_reg=False)
plt.show()

# Create a stacked histogram showing Out of State Tuition based on the Private column. 
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue='Private',palette='coolwarm')
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
plt.show()

# Create a similar histogram for the Grad.Rate column.
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
plt.show()

# There seems to be a private school with graduation rate higher than 100%
df[df['Grad.Rate'] > 100]
# we will set the Grad Rate for this to be 100
df['Grad.Rate']['Cazenovia College'] = 100

# We will re-do the facet grid to see, whether it went through
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
plt.show()

'''
KMeans clustering 
'''
from sklearn.cluster import KMeans

# We will create an instance with two clusters (private and public)
kmeans = KMeans(n_clusters=2)

# Now we will fit it to the entire set of features except the Private label
kmeans.fit(df.drop('Private',axis=1))

# Lets check the cluster centers
kmeans.cluster_centers_

# see the cluster labels assigned
kmeans.labels_

'''
Evaluation
There is no perfect way to evaluate clustering if you don't have the labels, 
however since this is just an exercise, we do have the labels, so we take advantage 
of this to evaluate our clusters, keep in mind, you usually won't have this luxury 
in the real world.

** Create a new column for df called 'Cluster', which is a 1 for a Private school, 
and a 0 for a public school.**
'''

# Create a new column for df called 'Cluster', which is a 1 for a Private school, 
# and a 0 for a public school

def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

# Do the sample evaluation
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

from sklearn.metrics.cluster import homogeneity_score
homogeneity_score(df['Cluster'],kmeans.labels_)