# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 03
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import sys
from sklearn.metrics import silhouette_score
from scipy import arange
    

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# read dataset
filename = sys.argv[1]
data = pd.read_csv(filename, header=None)

epsRange = arange(0.1, 1.1, 0.1)
minSamples = range(2, 11)
silhouetteCoef = []
n_clusters = []
colors = ['black', 'red', 'gray', 'darkgreen', 'lightcoral', 'royalblue', 'hotpink', 'lightseagreen', 'darkorange']

# loop to dbscan
for j, minSample in enumerate(minSamples):
    print("minSample: ", minSample)
    for e in epsRange:
        print("eps: ", e)
        db = DBSCAN()
        #db = DBSCAN(eps=e, min_samples=minSample)
        db.fit(data)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))

        # silhouette coeficient score
        silhouetteCoef.append(silhouette_score(data, labels))

    # plot
    plt.scatter(epsRange, silhouetteCoef, color=colors[j], label="min_samples=" + str(minSample))

    for i, n_cluster in enumerate(n_clusters):
        plt.annotate(n_cluster, (epsRange[i], silhouetteCoef[i]))
        
    plt.plot()  
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", fancybox=True, shadow=True)
    plt.xlabel("eps")
    plt.xticks(epsRange)
    plt.ylabel("Silhouette Coeficient")

    silhouetteCoef = []
    n_clusters = []

plt.savefig("DBSCAN.png", bbox_inches='tight')