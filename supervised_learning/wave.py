import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

#mglearn.plots.plot_knn_regression(n_neighbors=3)
#plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)

# split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to 3
reg = KNeighborsRegressor(n_neighbors=3)
# train the model using the training datasets
reg.fit(X_train, y_train)

# predict
print("Test set predictions: \n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.5f}".format(reg.score(X_test, y_test)))


def show(X, y):
	plt.plot(X, y, 'o')
	plt.ylim(-3, 3)
	plt.xlabel("Feature")
	plt.ylabel("Target")
	plt.show()


def analyze(X_train, X_test, y_train, y_test):		
	# analyze
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	# create 1,000 data points, evenly spaced between -3 and 3
	line = np.linspace(-3, 3, 1000).reshape(-1, 1)
	for n_neighbors, ax in zip([1, 3, 9], axes):
		# make predictions using 1, 3 or 9 neighbors
		reg = KNeighborsRegressor(n_neighbors=n_neighbors)
		reg.fit(X_train, y_train)
		ax.plot(line, reg.predict(line))
		ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
		ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
		train_score = reg.score(X_train, y_train)
		test_score = reg.score(X_test, y_test)
		ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, train_score, test_score))
		ax.set_xlabel("Feature")
		ax.set_ylabel("Target")

	axes[0].legend(["Model Predictions", "Training data/target", "Test data/target"], loc="best")
	plt.show()


def regression():
	X, y = mglearn.datasets.make_wave(n_samples=60)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	lr = LinearRegression().fit(X_train, y_train)
	print("lr.coef: {}".format(lr.coef_))
	print("lf.intercept: {}".format(lr.intercept_))
	train_score = lr.score(X_train, y_train)
	test_score = lr.score(X_test, y_test)
	print("Training score: {:.2f}".format(train_score))
	print("Testing score: {:.2f}".format(test_score))

show(X, y)
analyze(X_train, X_test, y_train, y_test)

mglearn.plots.plot_linear_regression_wave()
plt.show()

regression()
print("X shape: {}".format(X.shape))
print("y: {}".format(y))