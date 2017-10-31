# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:34:51 2017

@author: PulkitMaloo
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



##### Plot Dataset
def plotFunction(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker='o', alpha=0.5, linewidths=1, s=10)
    plt.show()

 #optimize - LDORC likhna h 


########### Algorithm #########################################################

def QDORC(df, abselon, eta):
    P = df.copy()      
    n = len(df)

    from scipy.spatial.distance import squareform, pdist, cdist
    w = pd.DataFrame(squareform(pdist(df)))    
    
    h = pd.DataFrame(w).applymap(lambda x: 1 if (x <= abselon) else 0)
    
    X = pd.DataFrame(np.eye(len(P), dtype=int))
    y = h.sum(axis=1).apply(lambda x: 1 if x>=eta else x/eta)
    
    core_index = list()
    border_index = list()
    noise_index = list()
    
    for index, val in enumerate(y):
        if val == 1:
            core_index.append(index)   
    core_count = len(core_index)
    print("Core", core_count)
    
    for index, val in enumerate(y):
        if val<1:
            if core_index and h.iloc[index,:][h.iloc[index,:] == 1].index[0] in core_index:
                border_index.append(index)
    border_count = len(border_index)
    print("Border", border_count)
    
    noise_index = list(set(range(n)) - set(core_index + border_index))
    noise_count = len(noise_index)    
    print("Noise", noise_count)
    
    
    while (noise_index):
        j_index = np.argmax(y[y<1])
        
        if len(noise_index) >= int((1 - y[j_index])*eta):
            for it in range(int((1 - y[j_index])*eta)):         
                 i_index = np.argmin(w.iloc[noise_index, j_index])
                 
                 X.iloc[i_index, i_index] = 0
                 X.iloc[i_index, j_index] = 1
                 noise_index.remove(i_index)

                 P.iloc[i_index, :] = P.iloc[j_index, :]
#                 print(count, "p"+str(i_index), "->", "p"+str(j_index))
                 noise_count -= 1             
                 core_count += 1
            y[j_index] = 1
            
            k_index = np.where(h.iloc[j_index, :] == 1)
            
            try:            
                for r in j_index + k_index:       
                    noise_index.remove(r)
                    
                    P.iloc[r, :] = P.iloc[j_index, :]
#                    print(count, "p"+str(r), "->", "p"+str(j_index))
                    noise_count -= 1
            except:
                pass
            
        else:
            for i_index in noise_index:
                k_index = np.argmin(w.iloc[i_index, list(set(range(len(P))) - set(noise_index))])
                
                X.iloc[i_index, i_index] = 0
                X.iloc[i_index, k_index] = 1
                noise_index.remove(i_index)

                P.iloc[i_index, :] = P.iloc[k_index, :]                
#                print(count, "p"+str(i_index), "->", "p"+str(k_index))
                noise_count -= 1
    
    #Repairing Accuracy
    from math import sqrt
    repair = sqrt((cdist(df, P).trace()) / n)
    print("Repairing Accuracy", round(repair, 2))
    
#    plt.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.5)
#    plt.show()           
#    plt.scatter(P.iloc[:,0], P.iloc[:,1], alpha=0.5)
#    plt.show()
    df.corrwith(P, axis = 1)
    return P

###############################################################################   





