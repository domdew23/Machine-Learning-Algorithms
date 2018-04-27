# Two types of supervised learning:
# Classification - the goal is to predict a class label, which is a choice from a predifined list of possibilites
# Regression - the goal is to predict a continous number, predicted value is an amount

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def KNeighbors():
	fig, axes = plt.subplots(1, 3, figsize=(10, 3))

	for n_neighbors, ax in zip([1, 3, 9], axes):
		# the fit method dreturns the object self, so we can instantiate and fit in one line
		clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
		mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
		mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
		ax.set_title("{} neighbor(s)".format(n_neighbors))
		ax.set_xlabel("Feature 0")
		ax.set_ylabel("Feature 1")
	axes[0].legend(loc=3)
	plt.show()

def show():
	# Plot dataset
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
	plt.legend(["Class 0", "Class 1"], loc=4)
	plt.xlabel("First feature")
	plt.ylabel("Second feature")
	plt.show()
	mglearn.plots.plot_knn_classification(n_neighbors=5)
	plt.show()

def svm():
	X, y = mglearn.tools.make_handcrafted_dataset()
	svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
	mglearn.plots.plot_2d_separator(svm, X, eps=.5)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
	# plot support vectors
	sv = svm.support_vectors_
	# class labels of support vectors are given by the sign of the dual coeffiecients
	sv_labels = svm.dual_coef_.ravel() > 0
	mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
	plt.xlabel("Feature 0")
	plt.ylabel("Feature 1")
	plt.show()


def vary_svm():
	fig, axes = plt.subplots(3, 3, figsize=(15, 10))

	for ax, C in zip(axes, [-1, 0, 3]):
		for a, gamma in zip(ax, range(-1, 2)):
			mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

	axes[0, 0].legend(["Class 0", "Class 1", "SV Class 0", "SV Class 1"], ncol=4, loc=(.9, 1.2))
	plt.show()


# Generate dataset
X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))
print("y: {}".format(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

#svm()
vary_svm()

#print("{}".format(mglearn.datasets.make_forge()))