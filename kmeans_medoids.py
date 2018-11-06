# coding: utf-8

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
dataname = sys.argv[1]
data = pd.read_csv(dataname, header=None)

# read headlines
textname = sys.argv[2]
with open(textname, encoding="utf-8") as f:
    content = f.readlines()
content = [x.strip() for x in content]
content = content[1:]

# number of centroids
krange = range(1, 30, 5)

# Dimensionality reduction
# pca = PCA(.001)
# data = pca.fit_transform(data)

# main loop to calculate k-means for all the centroids
scores = []
clusters = {}
labels, centers = None, None
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
    centers = km.cluster_centers_

# Elbow curve
#plt.plot(krange, scores)
#plt.xlabel("Número de clusters")
#plt.ylabel("SSE")
#plt.savefig("kmeansP_20_pca_20.png")

# separate items on their clusters
clusters_temp = {}
for item in range(len(labels)):
    if labels[item] in clusters_temp:
        clusters_temp[labels[item]].append(item)
    else:
        clusters_temp[labels[item]] = [item]
clusters = clusters_temp

outputFile = "output_medoids.txt"

# print a few headlines of each cluster
print(file=open(outputFile, "w", encoding="utf-8"))

n, m = np.shape(centers)
for i in clusters.keys(): # for each cluster
    centroid = centers[i] # centroid of a cluster
    dist2med = []
    dist2cent = []

    clusterMembers = data.iloc[clusters[i]].values # TODO: not working
    distance = euclidean_distances(clusterMembers, [centroid])
    distance = dict(zip(clusters[i], distance))

    # compute distance from element to centroid
    for k in clusters[i]: # for each element of a cluster
        d = distance[k]
        dist2cent.append([k, d])

    # order dist2cent by d
    dist2cent = np.array(dist2cent)
    dist2cent[dist2cent.argsort(axis=1)]

    # compute distance from element to medoid
    medoid = dist2cent[0]
    for k in clusters[i]:
        if k != medoid[0]:
            d0 = distance[k] * distance[k]
            d1 = medoid[1] * medoid[1]
            d = np.sqrt(d0 + d1)
            dist2med.append([k, d])

    # order dist2med by d
    dist2med = np.array(dist2med)
    dist2med[dist2med.argsort(axis=1)]

    print('Cluster ' + str(i), file=open(outputFile, "a", encoding="utf-8"))
    print('Medoid:', file=open(outputFile, "a", encoding="utf-8"))
    print(content[int(medoid[0])], file=open(outputFile, "a", encoding="utf-8"))
    print('Nearest Observations to Medoid:', file=open(outputFile, "a", encoding="utf-8"))
    for k in range(4):
        ind = int(dist2med[k][0])
        print('[' + str(ind) + '] ' + content[ind], file=open(outputFile, "a", encoding="utf-8"))
    print(file=open(outputFile, "a", encoding="utf-8"))
    print(file=open(outputFile, "a", encoding="utf-8"))

# print a few headlines of each cluster
outputFile = "output.txt"
print(file=open(outputFile, "w", encoding="utf-8"))
for i in clusters.keys():
    for _ in range(5):
        ind = random.choice(clusters[i])
        print('[' + str(ind) + '] ' + content[ind], file=open(outputFile, "a", encoding="utf-8"))
    print(file=open(outputFile, "a", encoding="utf-8"))
    print(file=open(outputFile, "a", encoding="utf-8"))
