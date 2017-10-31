from QDORC import QDORC

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

headers = ['x', 'y', 'l'] 
X = pd.read_csv('CDR.csv', names=headers)

Y = X['l'].copy()
X.drop(labels=['l'], axis=1, inplace=True)

eps = 0.3
eta = 12

P = QDORC(X, eps, eta)

from sklearn.cluster import KMeans as npt
k_x = npt(n_clusters = 3) 
labels_k = k_x.fit_predict(X)
model = npt(n_clusters = 3)
labels_p_k = model.fit_predict(P)

def depend_cdr():
    return k_x,labels_k, model, labels_p_k