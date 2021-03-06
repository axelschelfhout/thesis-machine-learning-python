import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from random import randint
from itertools import cycle

from sklearn import cluster
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #
# Data preparation

big_file = 'fakenames50k.csv'
small_file = 'fakeprofiledataset.csv'

train_file = big_file
test_file = small_file

# The header is in the first row. By adding header=0 we tell pd.read_csv it is
train_df = pd.read_csv(train_file, header=0, encoding='utf-8-sig', engine='python')
test_df = pd.read_csv(test_file, header=0, encoding='utf-8-sig', engine='python')

# Used df.dtypes to see what the column names are.
# print(train_df.dtypes)
# df['GenderTarget'] = 4

# Move gender from the first position, to the last position
cols = train_df.columns.tolist()
cols = cols[2:] + cols[:2]
train_df = train_df[cols]
test_df = test_df[cols]

# Transform the gender STRING to integer. So we can work with it. Female: 0, Male: 1
train_df.Gender = train_df.Gender.map({'female': 0, 'male': 1}).astype(int)
test_df.Gender = test_df.Gender.map({'female': 0, 'male': 1}).astype(int)

# Remove all the other Object types in the data frame.
df_drop = train_df.dtypes[train_df.dtypes.map(lambda x: x == 'object')].keys()
train_df = train_df.drop(df_drop, axis=1)
dropable_columns = ['NationalID', 'Longitude', 'Latitude', 'WesternUnionMTCN', 'MoneyGramMTCN', 'TelephoneCountryCode', 'CCNumber', 'CVV2', 'Number']
train_df = train_df.drop(dropable_columns, axis=1)

test_df = test_df.drop(df_drop, axis=1)
test_df = test_df.drop(dropable_columns, axis=1)

evaluate_df = test_df
# evaluate_df.Gender = evaluate_df.Gender.map({'female': 0, 'male': 1}).astype(int)
# test_df = test_df.drop(['Gender'], axis=1)

print(train_df.describe())

# The data frames values function will make this in to a Numpy array so we can use it in sklearn.
train_data = train_df.values
test_data = test_df.values
evaluate_data = evaluate_df.values
print(type(train_data))
# From https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii


# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #
# KMeans clustering
# For testing, set the targets
n_targets = 5

data = test_data
# data = scale(data)

np.random.shuffle(data)

# Make SKLEARN try to calculate the amount of clusters.
# FROM : http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html
print("How many clusters do we need?")
calc_clusters = cluster.AffinityPropagation().fit(data)
# print(calc_clusters)

cluster_centers_indices = calc_clusters.cluster_centers_indices_
print("We need " + str(len(cluster_centers_indices)))

labels = calc_clusters.labels_
n_clusters_ = len(cluster_centers_indices)

# Create plot with clusters
# plt.figure(1)
# plt.clf()
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = data[cluster_centers_indices[k]]
#     plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in data[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# from http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html

### --- ### --- ### --- ### --- ### --- ###

n_targets = n_clusters_

k_means = cluster.KMeans(n_clusters=n_targets)
k_means.fit(data)

print(k_means.labels_)


exit()

