# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:31:26 2017

@author: PulkitMaloo
"""
import numpy as np
import matplotlib.pyplot as plt

width = 0.1
kink = 0.3
fig = plt.figure()
ax = fig.gca()

iris_db = 0.669771734606
iris_qdorc = 0.758205727819

gis_db = 0.906654928
gis_qdorc = 0.944734875561

plt_db = ax.bar(0.05, iris_db, width, color = 'y')
plt_qdorc = ax.bar(0.05+width+0.007, iris_qdorc, width)
plt_db_1 = ax.bar(kink, gis_db, width, color = 'y')
plt_qdorc_1 = ax.bar(kink+width+0.007, gis_qdorc, width)

ax.set_ylabel('NMI')
ax.set_xticks((0.15+0.007, 0.4+0.007))
ax.set_xticklabels(('Iris', 'CDR'))
ax.set_yticks(np.array((0,1,2,3,4,5,6,7,8,9,10))/10)
ax.legend((plt_db, plt_qdorc), ('DBSCAN', 'DBRC'))

plt.xlim(0,0.8)
plt.grid(False)
plt.show()