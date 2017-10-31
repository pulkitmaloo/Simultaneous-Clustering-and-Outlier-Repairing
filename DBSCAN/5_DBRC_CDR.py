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

data = pd.DataFrame({'x':[3,5,8,1], 'y':[0,0,0,0]})


headers = ['x', 'y', 'l'] 
X = pd.read_csv('CDR.csv', names=headers)

Y = X['l'].copy()
X.drop(labels=['l'], axis=1, inplace=True)


#####  Plot Dataset ###########################################################


#
#plt.scatter(X.iloc[:,0],X.iloc[:,1],c=Y)
#plt.xlim(4,8)
#plt.show()
#
#plt.scatter(P.iloc[:,0],P.iloc[:,1],c=Y)
#plt.xlim(4,8)
#plt.show()
#

###############################################################################


eps = 0.3
eta = 12

print("eps", eps, "eta", eta)

P = QDORC(X, eps, eta)# float(input("Input eps : ")), int(input("Input eta : ")))


from sklearn.cluster import DBSCAN
db_x = DBSCAN(eps = eps, min_samples = eta)
labels_db = db_x.fit_predict(X)
print("DBSCAN clusters =", len(set(labels_db)) - (1 if -1 in labels_db else 0))
from depend_5 import depend_cdr
k_x,labels_k, model, labels_p_k = depend_cdr()
db_p = DBSCAN(eps = eps, min_samples = eta)
labels_p_db = db_p.fit_predict(P)
print("DBRC  clusters =", len(set(labels_p_db)) - (1 if -1 in labels_p_db else 0))


from sklearn.metrics import normalized_mutual_info_score

nmi_db = normalized_mutual_info_score(Y, labels_db)
nmi_k = normalized_mutual_info_score(Y, labels_k)
nmi_p_db = normalized_mutual_info_score(Y, labels_p_db)
nmi_p_k = normalized_mutual_info_score(Y, labels_p_k)


###############################################################################


plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Y)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)

centroids_1 = k_x.cluster_centers_

for i in range(len(centroids_1)):
    plt.plot(centroids_1[i][0], centroids_1[i][1], marker='x', 
             markersize=10, markeredgecolor='black', markeredgewidth=3)

plt.show()

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels_db)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()


##### After

plt.scatter(P.iloc[:,0], P.iloc[:,1], c=labels_p_db)
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
#plt.show()

centroids_2 = model.cluster_centers_

for i in range(len(centroids_2)):
    plt.plot(centroids_2[i][0], centroids_2[i][1], marker='x', 
             markersize=10, markeredgecolor='black', markeredgewidth=3)

plt.show()

###############################################################################################################################################################

kink = 0.1
width = 0.1

fig = plt.figure()
ax = fig.gca()


plt_db = ax.bar(kink, nmi_db, width, color = 'y')

plt_qdorc = ax.bar(kink+width+0.007, nmi_k, width)

ax.set_xlabel('Algorithm')
ax.set_ylabel('NMI')

ax.set_xticks((kink+width/2, kink+width+0.01+width/2))

ax.set_xticklabels(('DBSCAN', 'DBRC'))
ax.set_yticks(np.array((0,1,2,3,4,5,6,7,8,9,10))/10)

ax.legend((plt_db, plt_qdorc), ('DBSCAN', 'DBRC'))

plt.xlim(0,0.5)
plt.show()




