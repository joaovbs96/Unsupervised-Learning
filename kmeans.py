# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 03
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
dataname = sys.argv[1]
data = pd.read_csv(dataname, header=None)

# number of centroids
krange = np.array(range(1, 30, 5))

# main loop to calculate k-means for all the centroids
scores = []
clusters = {}
labels = None
for k in krange:
    km = KMeans(n_clusters=k)         # Number of centroids

    # fit model to dataset
    km.fit(data)
    #scores.append(km.inertia_ / data.shape[0])
    scores.append(sum(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    print(str(k) + ': ' + str(km.inertia_))

    # run model on dataset
    labels = km.predict(data)

# Elbow curve
plt.plot(krange, scores)
plt.xlabel("Número de clusters")
plt.ylabel("SSE")
plt.savefig("kmeans.png")

    
