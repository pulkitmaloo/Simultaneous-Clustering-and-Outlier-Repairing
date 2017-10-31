# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:28:52 2017

@author: PulkitMaloo
"""

from QDORC import QDORC

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)  # we only take the first two features.
Y = iris.target

eps = 0.6
eta = 4

P = QDORC(X, eps, eta)

from sklearn.cluster import KMeans as npt
k_x = npt(n_clusters = 3) 
labels_k = k_x.fit_predict(X)
model = npt(n_clusters = 3)
labels_p_k = model.fit_predict(P)

def depend_iris():
    return k_x,labels_k, model, labels_p_k