from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

print(digits.data[-1:])
exit()


d = clf.predict(digits.data[-1:])
print(d)

# from sklearn.datasets.samples_generator import make_blobs
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
#                             random_state=0)
#
# print(X)
# print(labels_true)
# exit()