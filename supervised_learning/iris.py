from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression

import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

def print_info():
	print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

	print("Target names: {}".format(iris_dataset['target_names']))
	print("Target names: {}".format(iris_dataset['feature_names']))
	print("Type of data: {}".format(type(iris_dataset['data'])))
	print("Shape of data: {}".format(iris_dataset['data'].shape))

	# 0 - setosa, 1 - versicolor, 2 - virginica
	print("First five columns of data: \n{}".format(iris_dataset['data'][:5]))
	print("Target: \n{}".format(iris_dataset['target']))	


def print_shapes():
	print("----------------------------------------")
	print("X_train shape: {}".format(X_train.shape))
	print("y_train shape: {}".format(y_train.shape))
	print("X_test shape: {}".format(X_test.shape))
	print("y_test shape: {}".format(y_test.shape))	


def k_neighbors():
	knn = KNeighborsClassifier(n_neighbors=1)

	# Train knn with fit() passing the training data as arguments
	knn.fit(X_train, y_train)

	# new data point (scikit learn always expects 2D arrays)
	X_new = np.array([[5, 2.9, 1, 0.2]])

	# Make a prediction
	prediction = knn.predict(X_new)
	print("Prediction: {}".format(prediction))
	print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

	# Compute accuracy
	y_pred = knn.predict(X_test)
	print("Test set predictions:\n{}".format(y_pred))
	print("Test set score: {:.5f}".format(np.mean(y_pred == y_test)))
	print("Test set score: {:.5f}".format(knn.score(X_test, y_test)))


def graph():
	# Create dataframe from data in X_train
	# Label the columns using the strings in iris_dataset.feature_names
	iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
	# Create a scatter matrix from the dataframe, color by y_train
	grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
	#plt.show()


def decision_tree():
	# Decision Tree
	tree = tree.DecisionTreeClassifier()
	tree.fit(X_train, y_train)
	tree_pred = tree.predict(X_test)
	print("Tree test set score: {:.5f}".format(tree.score(X_test, y_test)))	


def gradient_booster():
	gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
	gbrt.fit(X_train, y_train)
	print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
	# plot the first few entries of the descision function
	print("Decisioon function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
	print("Argmax of decision functions:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
	print("Predictions:\n{}".format(gbrt.predict(X_test)))

	# show first few entries of predict_proba
	print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
	# show that  sums accros rows are one
	print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
	
	print("Argmax of predicted probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
	print("Predictions:\n{}".format(gbrt.predict(X_test)))


def log_regression():
	logreg = LogisticRegression()

	# represent each target by its class name ein the iris dataset
	named_target = iris_dataset.target_names[y_train]
	logreg.fit(X_train, named_target)
	print("Unique classes in training data: {}".format(logreg.classes_))
	print("Predictions: {}".format(logreg.predict(X_test)[:10]))
	argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
	print("Argmax of decision function: {}".format(argmax_dec_func[:10]))
	print("Argmax combined with classes_ {}".format(logreg.classes_[argmax_dec_func][:10]))

iris_dataset = load_iris()

# Split data into two parts:
# Training data (set) - data used to build the machine learning model
# Testing data - rest of the data which is used to assess how well the model works (new data)
# 75% training data and 25% testing data split is a good rule of thumb
# Data is usually denoted with X
# Labels are denoted with y

# Function returns 4 NumPy arrays -> training data, test data, training labels, test labels
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#gradient_booster()
log_regression()
