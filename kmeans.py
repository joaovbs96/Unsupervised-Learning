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
krange = 2000

# main loop to calculate k-means for all the centroids
scores = []
clusters = {}
labels = None
for k in range(10, krange, 20):
    km = KMeans(n_clusters=k,         # Number of centroids
                init='k-means++',     # Centroids initialized as long as possible from each other
                max_iter=1000,         # Number of iterations - default 300
                n_init=10)             # Runs kmeans 10 times with different initialization

    # fit model to dataset
    km.fit(data)
    scores.append(np.average(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)))
    print(str(k) + ': ' + str(km.inertia_))

    # run model on dataset
    labels = km.predict(data)

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

# Elbow curve
plt.plot(scores)
plt.xlabel("K")
plt.ylabel("Inertia Score")
plt.show()
