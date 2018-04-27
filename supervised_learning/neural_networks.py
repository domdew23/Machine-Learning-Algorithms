import mglearn
import matplotlib.pyplot as plt
import graphviz
import numpy as np
# linear regressor predictor:
# y = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b

# y is a weighted sum of the input features x[0] to x[p], weighted by learned coefficents w[0] to w[p]

#graph = mglearn.plots.plot_logistic_regression_graph()
#graph.render(view=True)
#graph = mglearn.plots.plot_single_hidden_layer_graph()
#graph.render(view=True)
#graph = mglearn.plots.plot_two_hidden_layer_graph()
#graph.render('two_hidden_layers.gv', view=True)

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.show()

# equation:
# h[0] = tanh(w[0,0] * x[0] + w[1,0] * x[1] + w[2,0] * x[2] + ... + w[p, 0] * x[p])
# h[1] = tanh(w[0,0] * x[0] + w[1,0] * x[1] + w[2,0] * x[2] + ... + w[p, 0] * x[p])
#.
#.
#.
# h[n] = tanh(w[0,0] * x[0] + w[1,0] * x[1] + w[2,0] * x[2] + ... + w[p, 0] * x[p])
# y = v[0] * h[0] + v[1] * h[1] + ... v[n] * h[n]

# where w are the weights between input x and hidden layer h
# v are the weights between hidden layer h and output y
# x are the input features
# h are intermediate computations
# the weights v and w are learned from data
# y is the computed output