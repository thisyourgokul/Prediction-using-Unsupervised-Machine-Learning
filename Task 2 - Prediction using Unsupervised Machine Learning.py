#!/usr/bin/env python
# coding: utf-8

# TASK 2 : Prediction using Unsupervised Machine Learning. From the given 'Iris' dataset, predict the optimum number of clusters and represent it visually.
# 
# LEVEL : Beginner
# 
# DATASET USED : https://bit.ly/3kXTdox
# 
# NAME OF THE AUTHOR : Gokul Raj, Data Science and Business Analytics Intern, The Sparks Foundation.

# In[1]:


#Import all required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

#Load the iris dataset
iris=datasets.load_iris()
iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
#You can see first 5 rows
iris_df.head()



# In[3]:


#Finding the optimum number of clusters for k-means classification
x=iris_df.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph,
#Allowing us to observe 'The elbow'
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#within cluster sum of squares
plt.show()


# In[11]:


#Creating the kmeans classifier
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)


# In[12]:


#Visualizing the clusters on the first two columns
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.legend()

