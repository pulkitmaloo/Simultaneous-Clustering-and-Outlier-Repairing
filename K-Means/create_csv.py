# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:46:18 2016

@author: PulkitMaloo
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.datasets import make_gaussian_quantiles

def makeGaussian(x, y, num, rand):
    return make_gaussian_quantiles( mean=(x, y), cov=0.2,                                   
                                   n_samples=num, n_features=2, 
                                   n_classes=2, random_state=rand )


X1, y1 = makeGaussian(62.7, 29.45, 300, 1)    
X2, y2 = makeGaussian(62, 31, 100, 2)
X3, y3 = makeGaussian(61.5, 29.5, 70, 2)
X4, y4 = makeGaussian(61.25, 28.5, 350, 2)
X5, y5 = makeGaussian(63.10, 27.9, 450, 2)


df = pd.DataFrame(X1)
df = df.append(pd.DataFrame(X2))
df = df.append(pd.DataFrame(X3))
df = df.append(pd.DataFrame(X4))
df = df.append(pd.DataFrame(X5))

print(df)

df.reset_index(drop=True, inplace=True)

df.to_csv('joensuu.csv', index_label = False, index = False, header = False)

plt.scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.3)
#Update joensuu.csv and remove index column







