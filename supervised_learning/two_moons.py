from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import mglearn
import matplotlib.pyplot as plt

def scores(model, model_name):
	train_score = model.score(X_train,  y_train)
	test_score = model.score(X_test, y_test)
	print(model_name + " Training set score: {:.3f}".format(train_score))
	print(model_name + " Test set score: {:.3f}".format(test_score))


def forest_graph():
	# trees in forest are stored in estimator_ attribute
	forest = RandomForestClassifier(n_estimators=5, random_state=2)
	forest.fit(X_train, y_train)

	fig, axes = plt.subplots(2, 3, figsize=(20,10))
	for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
		ax.set_title("Tree {}".format(i))
		mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

	mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
	axes[-1, -1].set_title("Random Forest")
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
	plt.show()


def neural_network():
	mlp = MLPClassifier(solver='lbfgs', random_state=0, activation='tanh', hidden_layer_sizes=[10, 10]).fit(X_train, y_train)
	mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
	plt.xlabel("Feature 0")
	plt.ylabel("Feature 1")
	plt.show()

def show_variation():
	fig, axes = plt.subplots(2, 4, figsize=(20,8))
	for axx, n_hidden_nodes in zip(axes, [10, 100]):
		for ax, alpha in zip(axx, [0.001, 0.01, 0.1, 1]):
			mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
			mlp.fit(X_train, y_train)
			scores(mlp, 'mlp n_hidden = [{}, {}] and alpha = {}'.format(n_hidden_nodes, n_hidden_nodes, alpha))
			mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
			mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
			ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
	plt.show()


def show_randomness():
	fig, axes = plt.subplots(2, 4, figsize=(20,8))
	for i, ax in enumerate(axes.ravel()):
		mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])
		mlp.fit(X_train, y_train)
		scores(mlp, 'mlp random_state = {}'.format(i))
		mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
		mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
	plt.show()

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#neural_network()
#show_variation()
show_randomness()

# ways to control neural network
# number of hidden layers, number of units(nodes) in each hidden layer, and regularization(alpha)