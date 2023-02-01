# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:31:26 2022

@author: Vivek
"""

# K-Means Clustering

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("D:\Machine Learning\dataset_files/Mall_Customers.csv")

X = data.iloc[:,[3,4]].values

sns.pairplot(data, palette='hot', hue='Spending Score (1-100)')
sns.pairplot(data, palette='hot', hue='Annual Income (k$)')
# WCSS - Within Cluster Sum of Square

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=10)
    kmeans = kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)

# Here no of clusters are 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=10)
Y = kmeans.fit_predict(X)
#print(Y)

# Visualization
plt.figure(figsize=(8,8))
plt.scatter(X[Y== 0,0], X[Y== 0,1], s=50, c='red', label='Cluster 1')
plt.scatter(X[Y== 1,0], X[Y==1,1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[Y== 2,0], X[Y==2,1], s=50, c='green', label='Cluster 3')
plt.scatter(X[Y== 3,0], X[Y== 3,1], s=50, c='yellow',label='Cluster 4')
plt.scatter(X[Y== 4,0], X[Y== 4,1], s=50, c='pink', label='Cluster 5')
              
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')
plt.title("Customers group")
plt.xlabel("Annual Income")
plt.ylabel("Income")
plt.show()



     
     
     


