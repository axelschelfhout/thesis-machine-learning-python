import pandas as pd
import numpy as np
import time

from random import randint
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #
# Data preparation

script_start_time = time.time()  # Timer
dataprep_start_time = time.time()  # Timer
file = 'fakenames50k.csv'

# The header is in the first row. By adding header=0 we tell pd.read_csv it is
df = pd.read_csv(file, header=0, encoding='utf-8-sig', engine='python')

# Move gender from the first position, to the last position
cols = df.columns.tolist()
cols = cols[2:] + cols[:2]
df = df[cols]

# Transform the gender STRING to integer. So we can work with it. Female: 0, Male: 1
# df.Gender = df.Gender.map({'female': 0, 'male': 1}).astype(int)

# Remove all the other Object types in the data frame.
df_drop = df.dtypes[df.dtypes.map(lambda x: x == 'object')].keys()
df = df.drop(df_drop, axis=1)
dropable_columns = ['NationalID', 'Longitude', 'Latitude', 'WesternUnionMTCN', 'MoneyGramMTCN', 'TelephoneCountryCode', 'CCNumber', 'CVV2', 'Number']
df = df.drop(dropable_columns, axis=1)

print(df.describe())

# The data frames values function will make this in to a Numpy array so we can use it in sklearn.
df = df.values

print(df)

# Shuffle the data.
np.random.shuffle(df)

dataprep_end_time = time.time()  # Timer


# Get a kmeans DF and knn DF
kmeans_df, knn_df = train_test_split(df, train_size=0.08, random_state=randint(0,len(df)))

# Kmeans #
kmeans_clustercount_start_time = time.time()  # Timer
# Determine the amount of clusters based on the available data.
print("How many clusters do we need?")
calc_clusters = cluster.AffinityPropagation().fit(kmeans_df)

cluster_centers_indices = calc_clusters.cluster_centers_indices_
print("We need " + str(len(cluster_centers_indices)))

# Set the amount of clusters
n_clusters = len(cluster_centers_indices)

kmeans_start_time = time.time()  # Timer
# Start kMeans. Set the amount of clusters to the algorithm and fit the data to it.
k_means = cluster.KMeans(n_clusters=n_clusters, init='k-means++')
k_means.fit(kmeans_df)

# Get the cluster labels from the fitted data.
labels = k_means.labels_

# Apply the cluster labels to a pandas data frame
kmeans_labeled_data = pd.DataFrame(data=kmeans_df[0:,0:], index=kmeans_df[0:,0], columns=kmeans_df[0,0:])

kmeans_labeled_data['cluster'] = pd.Series(labels, index=kmeans_labeled_data.index)

kmeans_clustercount_end_time = time.time()  # Timer
kmeans_end_time = time.time()  # Timer

print(kmeans_labeled_data.describe)

# Set the pandas data frame to a numpy array to use it in the kNN algorithm
labeled_data = kmeans_labeled_data.values

knn_start_time = time.time()  # Timer

# Score kNN model accuracy
def score_knn_model_by_iterations(data, iterations, train_size, n_neighbors=5):
    model_score = []
    for i in range(0, iterations):
        rand = randint(0,len(data))
        x_train, x_test = train_test_split(data, train_size=train_size, random_state=rand)

        f_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        f_knn.fit(x_train[0::, 0:4], x_train[0::, 4])

        this_score = f_knn.score(x_test[0::, :4], x_test[0::, 4:5])
        model_score.append(this_score)
    return sum(model_score)/len(model_score)

score_model_accuracy = score_knn_model_by_iterations(labeled_data, 20, 0.7, 5)
print("kNN Accuracy score on clustered data: " + str(score_model_accuracy))

# Use 5 neighbour classification (k=5)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the data frame with the cluster labels
knn.fit(kmeans_df, k_means.labels_)

# Predict over the whole set.
print(knn.predict(knn_df))

knn_end_time = time.time()  # Timer
script_end_time = time.time()  # Timer

# For each row in the knn dataframe, predict the label
# for row in knn_df:
#     print(row)  # Print the row
#     print(knn.predict([row]))  # Print the label

print("Data prep time: " + str(dataprep_end_time - dataprep_start_time) + " seconds")
print("kMeans algorithm: " + str(kmeans_end_time - kmeans_start_time) + " seconds")
print("kMeans with the cluster determine (AP) : " + str(kmeans_clustercount_end_time - kmeans_clustercount_start_time) + " seconds")
print("kNN algorithm: " + str(knn_end_time - knn_start_time) + " seconds")
print("The whole script took : " + str(script_end_time - script_start_time) + " seconds to complete")

# EOF;
