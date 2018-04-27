from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
import mglearn

def transform(X_train, scaler):
	# transform data
	X_train_scaled =  scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	# print dataset properties before and after scaling
	print("Transformed shaped: {}".format(X_train_scaled.shape))
	print("Per-feature minimum before scaling: {}".format(X_train.min(axis=0)))
	print("Per-feature maximum before scaling: {}".format(X_train.max(axis=0)))
	print("Per-feature minimum after scaling: {}".format(X_train_scaled.min(axis=0)))
	print("Per-feature maximum after scaling: {}".format(X_train_scaled.max(axis=0)))


def scores(model, X_train, y_train, X_test, y_test, model_type=''):
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)
	print(model_type + " Training score: {:.2f}".format(train_score))
	print(model_type + " Testing score: {:.2f}".format(test_score))
	#print(model_type + " Number of features used: {}".format(np.sum(model.coef_ != 0)))
	print()


def graph_example():
	mglearn.plots.plot_pca_illustration()
	plt.show()


def graph():
	fig, axes = plt.subplots(15, 2, figsize=(10, 20))
	maligant = cancer.data[cancer.target == 0]
	bengin = cancer.data[cancer.target == 1]

	ax = axes.ravel()

	for i in range(30):
		_, bins = np.histogram(cancer.data[:, i], bins=50)
		ax[i].hist(maligant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
		ax[i].hist(bengin[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
		ax[i].set_title(cancer.feature_names[i])
		ax[i].set_yticks(())
	ax[0].set_xlabel("Feature magnitude")
	ax[0].set_ylabel("Frequency")
	ax[0].legend(["Maligant", "Bengin"], loc='best')
	fig.tight_layout()
	plt.show()



def scale():
	# preprocesing using 0-1 scaling
	scaler.fit(X_train)

	X_train_scaled =  scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	svm.fit(X_train_scaled, y_train)
	scores(svm, X_train_scaled, y_train, X_test_scaled, y_test, 'svm after 0-1 scaling')

	# preprocessing using standard scaler
	stan_scaler = StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	svm.fit(X_train_scaled, y_train)
	scores(svm, X_train_scaled, y_train, X_test_scaled, y_test, 'svm after standard scaling')



cancer = load_breast_cancer()
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

svm = SVC(C=100)
svm.fit(X_train, y_train)
scores(svm, X_train, y_train, X_test, y_test, 'svm before scaling')

print(X_train.shape)
print(X_test.shape)
#transform(X_train, scaler)
#graph_example()
graph()