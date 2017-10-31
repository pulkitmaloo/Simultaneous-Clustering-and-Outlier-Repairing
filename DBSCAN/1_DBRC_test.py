# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:34:51 2017

@author: PulkitMaloo
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

##### Load Dataset
## s3
#headers = ['x', 'y']
#df = pd.read_csv('s3.csv',  names = headers)
#df.columns = headers

df = pd.DataFrame({'x':[3,5,8,1], 'y':[0,0,0,0]})

###### joensuu
#headers = ['x', 'y']
#df = pd.read_csv('joensuu.csv',  names = headers)
#df.columns = headers

##### Plot Dataset
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker='o', alpha=0.5, linewidths=1, s=10)
#plt.show()

abselon = 2.5
eta = 3

P = pd.DataFrame()
P = P.append(df)
h = pd.DataFrame()

from scipy.spatial.distance import squareform, pdist
w = pd.DataFrame(squareform(pdist(P)))
h = pd.DataFrame(w).applymap(lambda x: 1 if x <= abselon else 0)

X = pd.DataFrame(np.eye(len(P), dtype=int))

y_eta = h.sum(axis=1)
y = y_eta.apply(lambda x: 1 if x>=eta else x/eta)

core_index = list()
border_index = list()
noise_index = list()

for index, val in enumerate(y):
    if val == 1:
        core_index.append(index)        
print("Core", len(core_index))

for index, val in enumerate(y):
    if val<1:
        if h.iloc[index,:][h.iloc[index,:] == 1].index[0] in core_index:
            border_index.append(index)
print("Border", len(border_index))

noise_index = list(set(range(len(P))) - set(core_index + border_index))
print("Noise", len(noise_index))

########### Algorithm #########################################################
count = 0
while (len(noise_index) > 0):
    j_index = np.argmax(y[y<1])
    if len(noise_index) >= int((1 - y[j_index])*eta):
        for it in range(int((1 - y[j_index])*eta)):         
             i_index = np.argmin(w.iloc[noise_index, j_index])
             X.iloc[i_index, i_index] = 0
             X.iloc[i_index, j_index] = 1
             noise_index.remove(i_index)
        y[j_index] = 1
        
        k_index = np.where(h.iloc[j_index,:] == 1)
        try:            
            for r in j_index + k_index:       
                noise_index.remove(r)
        except:
            pass
    else:
        for i_index in noise_index:
            k_index = np.argmin(w.iloc[i_index,list(set(range(len(df))) - set(noise_index))])            
            X.iloc[i_index, i_index] = 0
            X.iloc[i_index, k_index] = 1
            noise_index.remove(i_index)
    count += 1
#    print(count)
   
###############################################################################   
#print(P)    

for i in range(len(X)):
    for j in range(len(X)):
        if i!=j:    
            if X.iloc[i,j] == 1:
                P.iloc[i,:] = P.iloc[j,:]
                print("P"+ str(i)+" -> P"+str(j))



plt.scatter(df.iloc[:,0], df.iloc[:,1], s = 40, c='black')
plt.show()

plt.scatter(P.iloc[:,0], P.iloc[:,1], s=40, c='black')
plt.xlim(0, 9)
plt.show()