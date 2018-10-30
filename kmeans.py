import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import sys
from scipy.spatial.distance import cdist


# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

filename = sys.argv[1]
data = pd.read_csv(filename)

# number of centroids
krange = 16

# main loop to calculate k-means for all the centroids
scores = []
for k in range(1, krange):
    km = KMeans(n_clusters=k,         # Number of centroids
                init='k-means++',     # Centroids initialized as long as possible from each other
                max_iter=500,         # Number of iterations - default 300
                n_init=10)             # Runs kmeans 10 times with different initialization

    km.fit(data)
    #scores.append(km.inertia_)
    scores.append(np.average(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1)))
    #scores.append(-km.score(data))
    print(str(k) + ': ' + str(km.inertia_))


# Elbow curve
plt.plot([i for i in range(10)], scores)
plt.xlabel("K")
plt.xticks(krange)
plt.ylabel("Inertia Score")
plt.show()
