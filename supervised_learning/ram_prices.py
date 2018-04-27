import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def show():
	plt.semilogy(ram_prices.date, ram_prices.price)
	plt.xlabel("Year")
	plt.ylabel("Price in $/Megabyte")
	plt.show()


ram_prices = pd.read_csv("data/ram_price.csv")
# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# predict prices based on data
X_train = data_train.date[:, np.newaxis]
# we use log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# compare
plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree Prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear Prediction")
plt.legend()
plt.show()

# decision trees work well when you have features that are on completely diffeent scales
# or a mix of binary and continous features
# decision trees tend to overfit and provide poor generalization preformance

# Ensembles:
# random forests - a collection of decision trees, where each tree is slightly different from the others
#gradient boosted decision trees - learn from mistakes of previous trees, require careful tuning of parameters

# both models do not work well with high-dimensional sparse data
