# coding: utf-8

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
                #init='k-means++')     # Centroids initialized as long as possible from each other
                #max_iter=700,         # Number of iterations - default 300
                #n_init=5)            # Runs kmeans 10 times with different initialization

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
plt.savefig("kmeansP_20.png")

# separate items on their clusters
clusters_temp = {}
for item in range(len(labels)):
    if labels[item] in clusters_temp:
        clusters_temp[labels[item]].append(item)
    else:
        clusters_temp[labels[item]] = [item]
clusters = clusters_temp

# read headlines
textname = sys.argv[2]
with open(textname, encoding="utf-8") as f:
    content = f.readlines()
content = [x.strip() for x in content]

# print a few headlines of each cluster
print(file=open("output.txt", "w", encoding="utf-8"))
for i in clusters.keys():
    for _ in range(5):
        print(content[random.choice(clusters[i])], file=open("output.txt", "a", encoding="utf-8"))
    print(file=open("output.txt", "a", encoding="utf-8"))
    print(file=open("output.txt", "a", encoding="utf-8"))

    
