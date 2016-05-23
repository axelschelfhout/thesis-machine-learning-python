import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

filename = 'fakenames50k.csv'
csv_reader = pd.read_csv(filename, delimiter=',', dtype=None)
whole_data = csv_reader.as_matrix()  # turn it into numpy.ndarray > needed for sklearn

subset_size = 5000
subset = whole_data[:subset_size]
data = np.empty((subset_size, 3))

target = []

count = 0
for row in subset:
    gender = row[1]  # 1 - Gender
    age = row[22]  # 22 - Age
    kg = row[39]   # 39 - Kg
    cm = row[41]   # 41 - Cm

    data[count, 0] = age
    data[count, 1] = kg
    data[count, 2] = cm

    gender_int = 0
    if gender == 'female':
        gender_int = 1

    target.append(gender_int)
    count += 1

targets = np.unique(target)

print(target)
exit()
n_targets = len(targets)
data = scale(data)

n_samples, n_features = data.shape
print("n_targets: %d, n_samples: %d, n_features: %d"
      % (n_targets, n_samples, n_features))

# KMeans clustering
k_means = cluster.KMeans(init='k-means++', n_clusters=n_targets)
k_means.fit(data)




# print(k_means.score(data))
# exit()


print(metrics.homogeneity_score(target, k_means.labels_))
print(metrics.completeness_score(target, k_means.labels_))
print(metrics.v_measure_score(target, k_means.labels_))
print(metrics.adjusted_rand_score(target, k_means.labels_))
print(metrics.adjusted_mutual_info_score(target, k_means.labels_))
print(metrics.silhouette_score(data, k_means.labels_, metric='euclidean', sample_size=subset_size))


# ---------------------------------------------------------------------------------- #
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = cluster.KMeans(init='k-means++', n_clusters=n_targets, n_init=10)
kmeans.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the fake profile dataset (male/female) (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


print(78 * '_')

digits = datasets.load_digits()
ddata = scale(digits.data)
n_samples, n_features = ddata.shape
n_digits = len(np.unique(digits.target))
print("n_digits: %d, n_samples: %d, n_features: %d"
      % (n_digits, n_samples, n_features))
labels = digits.target
sample_size = 300
dig_k_means = cluster.KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
dig_k_means.fit(ddata)

print(metrics.homogeneity_score(labels, dig_k_means.labels_))
print(metrics.completeness_score(labels, dig_k_means.labels_))
print(metrics.v_measure_score(labels, dig_k_means.labels_))
print(metrics.adjusted_rand_score(labels, dig_k_means.labels_))
print(metrics.adjusted_mutual_info_score(labels, dig_k_means.labels_))
print(metrics.silhouette_score(ddata, dig_k_means.labels_, metric='euclidean', sample_size=subset_size))

