import time
start = time.time()

from sklearn.datasets import load_boston
from functions import print_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import mglearn


def print_scores(model, model_type=''):
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)
	print(model_type + " Training score: {:.2f}".format(train_score))
	print(model_type + " Testing score: {:.2f}".format(test_score))
	print(model_type + " Number of features used: {}".format(np.sum(model.coef_ != 0)))
	print()

def graph_ridge(ridge, ridge10, ridge01, lr):
	plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
	plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
	plt.plot(ridge01.coef_, 'v', label='Ridge alpha=.1')

	plt.plot(lr.coef_, 'o', label='Linear Regression')
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.hlines(0, 0, len(lr.coef_))
	plt.ylim(-25, 25)
	plt.legend()
	plt.show()

	mglearn.plots.plot_ridge_n_samples()
	plt.show()

def graph_lasso(lasso, lasso001, lasso00001, ridge01):
	plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
	plt.plot(lasso001.coef_, '^', label='Lasso alpha=.01')
	plt.plot(lasso00001.coef_, 'v', label='Lasso alpha=.0001')

	plt.plot(ridge01.coef_, 'o', label='Ridge alpha=.1')
	plt.legend(ncol=2, loc=(0, 1.05))
	plt.ylim(-25, 25)
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficent magnitude")
	plt.show()

def scores():
	print_scores(lr, 'lr')
	print_scores(ridge, 'ridge')
	print_scores(ridge10, 'ridge10')
	print_scores(ridge01, 'ridge01')
	print_scores(lasso, 'lasso')
	print_scores(lasso001, 'lasso001')
	print_scores(lasso00001, 'lasso00001')

boston = load_boston()
# print_dataset(boston)

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X train:{}".format(X_train))
print("X test:{}".format(X_test))
import sys
sys.exit()
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=.1).fit(X_train, y_train)
lasso = Lasso().fit(X_train, y_train)

# increase the default settting of 'max_iter', otherwise the model would warn us that we should increase it
lasso001 = Lasso(alpha=.01, max_iter=100000).fit(X_train, y_train)
lasso00001 = Lasso(alpha=.0001, max_iter=100000).fit(X_train, y_train)

#graph_ridge(ridge, ridge10, ridge01, lr)
#graph_lasso(lasso, lasso001, lasso00001, ridge01)

finish = time.time()
print("Took {:.2f} seconds".format(finish - start))



# Ridge is usually better unless you have a large amount of features and expect only a few to be important, then use Lasso

