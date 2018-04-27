from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals.six import StringIO
from sklearn.neural_network import MLPClassifier

from functions import print_dataset
from sklearn.svm import SVC

import pydotplus
import matplotlib.pyplot as plt
import numpy as np
import mglearn

def KNeighbors(X_train, X_test, y_train, y_test):
	training_accuracy = []
	test_accuracy = []
	# try n_neighbors from 1 to 10
	neighbors_settings = range(1, 11)

	for n_neighbors in neighbors_settings:
		# build model
		clf = KNeighborsClassifier(n_neighbors=n_neighbors)
		clf.fit(X_train, y_train)
		# record training set accuracy
		training_accuracy.append(clf.score(X_train, y_train))
		# record test set accuracy
		test_accuracy.append(clf.score(X_test, y_test))

	means = []
	for train, test in zip(training_accuracy, test_accuracy):
		mean = (train + test) / 2
		means.append(mean)

	print("Means: {}".format(means))

	plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
	plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("n_neighbors")
	plt.legend()
	plt.show()

def DecisionTree(depth=None):
	tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
	tree.fit(X_train, y_train)
	scores(tree, "Decision Tree, max_depth = {} ".format(depth))
	return tree

def scores(model, model_name):
	train_score = model.score(X_train,  y_train)
	test_score = model.score(X_test, y_test)
	print(model_name + " Training set score: {:.3f}".format(train_score))
	print(model_name + " Test set score: {:.3f}".format(test_score))


def graph(logreg, logreg100, logreg001, cancer):
	plt.plot(logreg.coef_.T, 'o', label='C=1')
	plt.plot(logreg100.coef_.T, '^', label='C=100')
	plt.plot(logreg001.coef_.T, 'v', label='C=.001')

	plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
	plt.hlines(0, 0, cancer.data.shape[1])
	plt.ylim(-5, 5)
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.legend()  
	plt.show()


def graph_L1():
	for C, marker in zip([0.001, 1, 100], ['o','^', 'v']):
		lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
		scores(lr_l1, "lr_l1 with C={}".format(C))
		plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

	plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
	plt.hlines(0, 0, cancer.data.shape[1])
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.ylim(-5, 5)
	plt.legend(loc=3)  
	plt.show()

def graph_tree(tree, tree_name):
	dot_data = StringIO()
	export_graphviz(tree, out_file=dot_data, feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True, rounded=True, impurity=False)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(tree_name + '.pdf')


def feature_importance(model):
	print("Feature importances: \n{}".format(model.feature_importances_))
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel("Feature Importance")
	plt.ylabel("Feature")
	plt.show()


def logistic_regression():
	logreg = LogisticRegression().fit(X_train, y_train)
	logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
	logreg001 = LogisticRegression(C=.001).fit(X_train, y_train)

	scores(logreg, 'logreg')
	scores(logreg100, 'logreg100')
	scores(logreg001, 'logreg001')
	#graph(logreg, logreg100, logreg001, cancer)
	#graph_L1()


def random_forest():
	forest = RandomForestClassifier(n_estimators=100, random_state=0)
	forest.fit(X_train, y_train)
	scores(forest, "forest")
	feature_importance(forest)


def decision_trees():
	tree_0 = DecisionTree()
	tree_4 = DecisionTree(4)

	feature_importance(tree_0)

	#graph_tree(tree_0, 'tree_0')
	#graph_tree(tree_4, 'tree_4')
	#tree = mglearn.plots.plot_tree_not_monotone()
	#plt.show()

def neural_network():
	# compute the mean value per feautre on training set
	mean_on_train = X_train.mean(axis=0)
	# compute standard deviation of each feature on training set
	std_on_train = X_train.std(axis=0)

	# subtract the mean and scale by inverse standard deviation
	# afterward mean = 0 and std = 1 
	X_train_scaled = (X_train - mean_on_train) / std_on_train
	# use same transformation on test set
	X_test_scaled = (X_test - mean_on_train) / std_on_train

	mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
	mlp.fit(X_train_scaled, y_train)

	train_score = mlp.score(X_train_scaled,  y_train)
	test_score = mlp.score(X_test_scaled, y_test)
	print('mlp scaled' + " Training set score: {:.3f}".format(train_score))
	print('mlp scaled' + " Test set score: {:.3f}".format(test_score))
	graph_nn(mlp)

def graph_nn(mlp):
	plt.figure(figsize=(20, 5))
	plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
	plt.yticks(range(30), cancer.feature_names)
	plt.xlabel("Columns in weight matrix")
	plt.ylabel("Input feature")
	plt.colorbar()
	plt.show()


def gradient_classifier():
	gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
	gbrt.fit(X_train, y_train)
	scores(gbrt, "gradient classifier")
	feature_importance(gbrt)



def svm_kernal(X, y, name):
	svc = SVC()
	svc.fit(X, y)
	scores(svc, name)


def graph_svm():
	plt.plot(X_train.min(axis=0), 'o', label='min')
	plt.plot(X_train.max(axis=0), '^', label='max')
	plt.legend(loc=4)
	plt.xlabel("Feature index")
	plt.ylabel("Feature Magnitude")
	plt.yscale("Log")
	plt.show()

def scale():
	# compute minimum value per feature on training set
	min_on_training = X_train.min(axis=0)
	# compute range of each feature on training set
	range_on_training = (X_train - min_on_training).max(axis=0)

	# subtract the min and divide by range
	# after min=0 and max=1 for each feature
	X_train_scaled = (X_train - min_on_training) / range_on_training
	print("Maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))
	print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
	X_test_scaled = (X_test - min_on_training) / range_on_training
	svc = SVC(C=1000)
	svc.fit(X_train_scaled, y_train)

	train_score = svc.score(X_train_scaled,  y_train)
	test_score = svc.score(X_test_scaled, y_test)
	print('svc scaled' + " Training set score: {:.3f}".format(train_score))
	print('svc scaled' + " Test set score: {:.3f}".format(test_score))

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

#logistic_regression()
#decision_trees()
#random_forest()
#gradient_classifier()
#svm_kernal(X_train, y_train, 'svc')
#scale()
neural_network()
#graph_svm()
#print_dataset(cancer)




