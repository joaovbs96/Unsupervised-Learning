# coding: utf-8
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
dataname = sys.argv[1]
data = pd.read_csv(dataname, header=None)

# number of centroids
best_n_clusters = 80

# dimensionality before pca
print("Dimensãoes antes de pca: ", data.shape[1])

# test a range of variances
variances = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# main loop to calculate k-means for all the centroids
scores = []
clusters = {}
labels = None
for variance in variances:
    pca = PCA(variance)
    data_pca = pca.fit_transform(data)

    print("Mantendo ", variance, " de variância, temos: ", data_pca.shape[1], " features")

    km = KMeans(n_clusters=best_n_clusters,      # Number of centroids
                init='k-means++',               # Centroids initialized as long as possible from each other
                max_iter=500,                   # Number of iterations - default 300
                n_init=10)                      # Runs kmeans 10 times with different initialization

    # fit model to dataset
    km.fit(data_pca)
    scores.append(km.inertia_ / data.shape[0])
    

    # run model on dataset
    #labels = km.predict(data)

# Elbow curve
plt.plot(variances, scores)
plt.xlabel("Variância")
plt.ylabel("SSE")
plt.xticks(variances)
plt.title("Kmeans para " + str(best_n_clusters) + " clusters utilizando PCA")
plt.savefig("pca.png")

# separate items on their clusters
#clusters_temp = {}
#for item in range(len(labels)):
#    if labels[item] in clusters_temp:
#        clusters_temp[labels[item]].append(item)
#    else:
#        clusters_temp[labels[item]] = [item]
#clusters = clusters_temp

# read headlines
#textname = sys.argv[2]
#with open(textname, encoding="utf-8") as f:
#    content = f.readlines()
#content = [x.strip() for x in content]

# print a few headlines of each cluster
#print(file=open("output.txt", "w", encoding="utf-8"))
#for i in clusters.keys():
#    for _ in range(5):
#        print(content[random.choice(clusters[i])], file=open("output.txt", "a", encoding="utf-8"))
#    print(file=open("output.txt", "a", encoding="utf-8"))
#    print(file=open("output.txt", "a", encoding="utf-8"))

    
