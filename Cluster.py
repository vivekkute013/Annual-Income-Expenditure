# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:46:12 2022

@author: Vivek
"""

''' Clustering KMeans '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Machine Learning\dataset_files/KMeans.csv')
X = dataset.iloc[:, [1 , 2]].values

''' Using elbow method to find the optimal number of clusters'''

from sklearn.cluster import KMeans
k = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', random_state = None)
    kmeans.fit(X)
    k.append(kmeans.inertia_)
plt.plot(range(1,11), k)
plt.title("Elbow Method")
plt.xlabel("No.of clusters")
plt.ylabel("k")
plt.show()

''' Fitting KMeans to dataset '''
kmeans = KMeans(n_clusters=5, init ='k-means++', random_state = None)
y_kmeans = kmeans.fit_predict(X)


'''Visualize the Cluster'''
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'pink', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'cyan', label = 'Centroid')
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k)")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()












