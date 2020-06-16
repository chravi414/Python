# Importing Numpy
import numpy as np

# creating a random array of floats
x = np.random.uniform(low=1, high=20, size=(20))
# print(x)

# Reshaping the array to 2-Dimensional
y = np.reshape(x, (4, 5))
print(y)
print("=================")

# Getting the max values of each row
indexes = np.arange(y.shape[0]), np.argmax(y, axis=1)
print(indexes)
y[indexes] = 0
print(y)
