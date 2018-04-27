# GaussianNB - can be applied to continuous data (non-integers), used on high-dimensional data
# BernoulliNB - assumes binary data (0 or 1)
# MultinomialNB - assumes count data (each feature represents an integer count)

import numpy as np

# BernoulliNB example:
# data
X = np.array([[0, 1, 0, 1],
			  [1, 0, 1, 1],
			  [0, 0, 0, 1],
			  [1, 0, 1, 0]])

# labels (targets)
y = np.array([0, 1, 0, 1]) 

# feature counts:
counts = {}
for label in np.unique(y):
	# iterate over each class
	# count (sum) entries of 1 per feature
	counts[label] = X[y==label].sum(axis=0)

print("Feature counts: \n{}".format(counts))