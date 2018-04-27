from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import mglearn


def graph(gbrt):
	fig, axes = plt.subplots(1, 2, figsize=(13, 5))
	mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
	scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4,  cm=mglearn.ReBl, function='predict_proba')

	for ax in axes:
		#plot training and test points
		mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
		mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)

		ax.set_xlabel("Feature 0")
		ax.set_ylabel("Feature 1")
	cbar = plt.colorbar(scores_image, ax=axes.tolist())
	axes[0].legend(["Test Class 0", "Test Class 1", "Train Class 0", "Train Class 1"], ncol=4, loc=(.1, 1.1))
	plt.show()	



X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y]

# can call train_test_split with arbritary number of arrays
# will be split in a constant manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)

# build gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

decision_function = gbrt.decision_function(X_test)

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(decision_function.shape))
print("Decision function: {}".format(decision_function[:6]))
print("Thresholded decision function: {}".format(decision_function > 0))
print("Predictions: {}".format(gbrt.predict(X_test)))

# make true and falses in 0 and 1
greater_zero = (decision_function > 0).astype(int)
# use 0 and 1 as indeces into classes
pred = gbrt.classes_[greater_zero]
# 3pred is the same as the output of gbrt.predict
print("Pred is equal to predictions: {}".format(np.all(pred == gbrt.predict(X_test))))
print("Decision minimum: {:.2f} || maximum: {:.2f}".format(np.min(decision_function), np.max(decision_function)))


print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
# show the first few entries of predict_proba
print("Predicted probabilities: \n{}".format(gbrt.predict_proba(X_test[:6])))
graph(gbrt)