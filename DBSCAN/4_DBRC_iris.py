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

###### joensuu


###############################################################################

from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)  # we only take the first two features.
Y = iris.target


###############################################################################

eps = 0.6
eta = 4

print("eps", eps, "eta", eta)

P = QDORC(X, eps, eta)# float(input("Input eps : ")), int(input("Input eta : ")))

###############################################################################



from sklearn.cluster import DBSCAN
db_x = DBSCAN(eps = eps, min_samples = eta)
labels_db = db_x.fit_predict(X)
print("DBSCAN clusters =", len(set(labels_db)) - (1 if -1 in labels_db else 0))
from depend_4 import depend_iris
k_x,labels_k, model, labels_p_k = depend_iris()
db_p = DBSCAN(eps = eps, min_samples = eta)
labels_p_db = db_p.fit_predict(P)
print("QDORC  clusters =", len(set(labels_p_db)) - (1 if -1 in labels_p_db else 0))


###############################################################################


from sklearn.metrics import normalized_mutual_info_score

nmi_db = normalized_mutual_info_score(Y, labels_db)
nmi_k = normalized_mutual_info_score(Y, labels_k)
nmi_p_db = normalized_mutual_info_score(Y, labels_p_db)
nmi_p_k = normalized_mutual_info_score(Y, labels_p_k)

###############################################################################


width = 0.1
kink = 0.1

fig = plt.figure()
ax = fig.gca()

plt_db = ax.bar(kink, nmi_db, width, color = 'y')

plt_qdorc = ax.bar(kink+width+0.007, kink+nmi_p_k, width)

ax.set_xlabel('Algorithm')

ax.set_xticks((kink+width/2, kink+width+0.01+width/2))
ax.set_xticklabels(('DBSCAN', 'DBRC'))

ax.set_ylabel('NMI')
ax.set_yticks(np.array((0,1,2,3,4,5,6,7,8,9,10))/10)

ax.set_title('UCI Iris Dataset')

ax.legend((plt_db, plt_qdorc), ('DBSCAN', 'DBRC'))

plt.xlim(0, 0.5)
plt.grid(False)
plt.show()

###############################################################################

