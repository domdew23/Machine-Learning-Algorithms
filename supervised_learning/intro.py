import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as display
import mglearn

from scipy import sparse

# numpy
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x\n{}".format(x))

# scipy
# Create 2D array with a diagonal of ones, and zeros everywhere else (Identity matrix)
eye = np.eye(4)
print("\nNumPy array:\n{}".format(eye))

# Convert the NumPy array to  SciPy sparse matrix is CSR format
# Only the non-zero entries are stored
spare_matrix = sparse.csr_matrix(eye)
print("\nScipy Sparse CSR matrix:\n{}".format(spare_matrix))
 
# COO format
data = np.ones(4)
row_indices = np.arange(4)
col_indices  = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("\nCOO representation:\n{}".format(eye_coo))

# matplotlib
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
plt.show()

# pandas
# Create a simple dataset of people
data = {
	'Name': ["John", "Anna", "Peter", "Linda"],
	'Location': ['New York', 'Paris', 'Berlin', 'London'],
	'Age': [24, 13, 53, 33]
}

data_pandas = pd.DataFrame(data)
# IPython.display allows 'pretty printing' of data frames
display.display(data_pandas)

# Select all rows that have an age column greater than 30
display.display(data_pandas[data_pandas.Age > 30])