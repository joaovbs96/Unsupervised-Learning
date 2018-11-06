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
from sklearn.metrics import silhouette_score

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
dataname = sys.argv[1]
data = pd.read_csv(dataname, header=None)

# number of centroids
best_n_cluster = 10

# Inicializations
clusters = {}
labels = None

# k-means execution
km = KMeans(n_clusters=best_n_cluster)         # Number of centroids
               
# fit model to dataset
km.fit(data)
    
# calculate error
score = sum(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]
print("Para", str(best_n_cluster), " clusters, erro:", str(score))

# run model on dataset
labels = km.predict(data)

# silhouette coeficient
silhouetteCoef = silhouette_score(data, labels)
print("Coeficiente de silhueta para K =" , str(best_n_cluster), "->", str(silhouetteCoef))

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

    
