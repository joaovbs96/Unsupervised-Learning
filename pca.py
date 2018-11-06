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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import metrics

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
dataname = sys.argv[1]
data = pd.read_csv(dataname, header=None)

# number of centroids
best_n_clusters = 4

# dimensionality before pca
print("Dimensões antes de pca: ", data.shape[1])

# test a range of variances
variances = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# main loop to calculate k-means for all the centroids
scores = []
silhouetteCoef = []
clusters = {}
labels = None
for variance in variances:
    pca = PCA(variance)
    data_pca = pca.fit_transform(data)

    print("Mantendo ", variance, " de variância, temos: ", data_pca.shape[1], " features")

    km = KMeans(n_clusters=best_n_clusters)      # Number of centroids

    # fit model to dataset
    km.fit(data_pca)
    km.predict(data_pca)
    labels = km.labels_

    scores.append(sum(np.min(cdist(data_pca, km.cluster_centers_, 'euclidean'), axis=1)) / data_pca.shape[0])
    silhouetteCoef.append(silhouette_score(data_pca, labels))
    

# Elbow curve
plt.plot(variances, scores)
plt.xlabel("Variância")
plt.ylabel("SSE")
plt.xticks(variances)
plt.title("Kmeans para " + str(best_n_clusters) + " clusters utilizando PCA")
plt.savefig("pcaSSE.png")

plt.clf()

# Silhouette coef
plt.plot(variances, silhouetteCoef)
plt.xlabel("Variância")
plt.ylabel("Coeficiente de silhueta")
plt.xticks(variances)
plt.title("Kmeans para " + str(best_n_clusters) + " clusters utilizando PCA")
plt.savefig("pcaCS.png")