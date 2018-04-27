import numpy as np

def print_dataset(dataset):
	try:
		print("Target names: {}".format(dataset['target_names']))
		print("Sample counts per class: \n{}".format({n: v for n, v in zip(dataset.target_names, np.bincount(dataset.target))}))
	except KeyError:
		pass
	print("Feature names: {}".format(dataset['feature_names']))
	print("Keys of dataset: \n{}".format(dataset.keys()))
	print("Type of data: {}".format(type(dataset['data'])))
	print("Shape of data: {}".format(dataset['data'].shape))
	print("First five columns of data: \n{}".format(dataset['data'][:5]))
	print("Target: \n{}".format(dataset['target']))

