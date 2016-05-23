from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pylab as pl

# filename = 'fakeprofiledataset.csv'
filename = 'fakenames50k.csv'
csv_reader = pd.read_csv(filename, delimiter=',', dtype=None)
whole_data = csv_reader.as_matrix()  # turn it into numpy.ndarray

subset = 1000
# create subset
data = whole_data[:subset]
print(data.shape)

# exit()

clustset = np.empty((subset,3))
count = 0
for row in data:
    gend = row[1]  # 1 - Gender
    age = row[22]  # 22 - Age
    kg = row[39]  # 39 - Kg
    cm = row[41]  # 41 - Cm

    clustset[count, 0] = age
    clustset[count, 1] = kg
    clustset[count, 2] = cm
    print("COUNT: "+str(count))
    count += 1


# k_range = range(1,14)
# k_means_var = [KMeans(n_clusters=k).fit(clustset) for k in k_range]
# centroids = [X.cluster_centers_ for X in k_means_var]
#
# # Calculate the Euclidean distance from
# # each point to each cluster center
# k_euclid = [cdist(clustset, cent, 'euclidean') for cent in centroids]
# dist = [np.min(ke,axis=1) for ke in k_euclid]
#
# # Total within-cluster sum of squeares
# wcss = [sum(d**2) for d in dist]
#
# # The toal sum of squares
# tss = sum(pdist(clustset)**2)/clustset.shape[0]
#
# # The between-cluster sum of squares
# bss = tss - wcss
#
# min = np.amin(bss)
# max = np.amin(bss)
#
# plt.plot(bss, 'bo')
# plt.show()


time_start = time.time()
k_mean = cluster.KMeans(n_clusters=8)

k_mean.fit(clustset)

t_batch = time.time() - time_start
print("Batch done in " + str(t_batch) + " seconds")

# print(kgemeen.labels_[::10])
# print(kgemeen.cluster_centers_)
unique_clust_labels = np.unique(k_mean.labels_)

print(unique_clust_labels)

time_end = time.time()

total_time = time_end - time_start
print("Time spend:" + str(total_time))

exit()

# Iris test set
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
print(k_means.labels_[::10])

