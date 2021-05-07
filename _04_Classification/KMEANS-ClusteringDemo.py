# -*- coding: utf-8 -*-
"""
K Means Clustering is an unsupervised learning algorithm that tries to cluster 
data based on their similarity. Unsupervised learning means that there is no 
outcome to be predicted, and the algorithm just tries to find patterns in the data. 
In k means clustering, we have the specify the number of clusters we want the data 
to be grouped into. The algorithm randomly assigns each observation to a cluster, 
and finds the centroid of each cluster. Then, the algorithm iterates 
through two steps: Reassign data points to the cluster whose centroid is closest. 
Calculate new centroid of each cluster. These two steps are repeated till the within 
cluster variation cannot be reduced any further. The within cluster variation is 
calculated as the sum of the euclidean distance between the data points and their 
respective cluster centroids.

@author: suvosmac
"""

# library imports
import matplotlib.pyplot as plt
import seaborn as sns

# use sci-kit learn we will generate some artificial data
from sklearn.datasets import make_blobs

# make_blobs will generate some gaussian blobs for clustering

# This returns a tuple and the first element is a numpy 2-d array
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)
data[0].shape

# lets plot this data
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()

'''
Perform clustering
'''
from sklearn.cluster import KMeans

# Initialize the KMeans object with the cluster number
kmeans = KMeans(n_clusters=4)
# Fit this on the features
kmeans.fit(data[0])

# If we want to see the cluster centers
kmeans.cluster_centers_
# see the cluster labels assigned
kmeans.labels_

# Now we will plot this out to see how it has done against our preassigned labels
# Keep in mind, that Kmeans is an unsupervised learning and the label data comparison will not
# exist.. this is just for reference

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()