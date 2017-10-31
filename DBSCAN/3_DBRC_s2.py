# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:30:37 2017

@author: PulkitMaloo
"""
from QDORC import QDORC

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


#####  Load Dataset ###########################################################
### s3
headers = ['x', 'y']
s3 = pd.read_csv('s2.csv',  names = headers, delim_whitespace=True)
s3.columns = headers

#####  Apply QDORC ############################################################

eps = 83000
eta = 180
print("eps", eps, "eta", eta)
P = QDORC(s3, eps, eta)

###### Plot Dataset ###########################################################
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 15)
labels_after = model.fit_predict(P)
##### Before
plt.scatter(s3.iloc[:,0], s3.iloc[:,1], alpha=0.5)
plt.xlim(0, 1000000)
plt.ylim(0, 1000000)
plt.show()           

#from sklearn.cluster import DBSCAN
#model = DBSCAN(eps, eta)
#labels_before = model.fit_predict(s3)
#print("Clusters", len(set(labels_before)) - (1 if -1 in labels_before else 0))

#unique_labels = set(labels_before)
##colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#plt.scatter(s3.iloc[:,0], s3.iloc[:,1], alpha= 0.5, c=labels_before)
#plt.xlim(0, 1000000)
#plt.ylim(0, 1000000)
#plt.show()


##### After
plt.scatter(P.iloc[:,0], P.iloc[:,1], alpha=0.5)
plt.xlim(0, 1000000)
plt.ylim(0, 1000000)
plt.show()




#from sklearn.cluster import DBSCAN
#model = DBSCAN(eps, 320)
#labels_after = model.fit_predict(P)
#print("Clusters", len(set(labels_after)) - (1 if -1 in labels_after else 0))

plt.scatter(P.iloc[:,0], P.iloc[:,1], alpha=0.5, c=labels_after)
plt.xlim(0, 1000000)
plt.ylim(0, 1000000)



centroids = model.cluster_centers_
for i in range(len(centroids)):
    plt.plot(centroids[i][0], centroids[i][1], marker='x', markersize=10, markeredgecolor='black', markeredgewidth=3)

plt.show()