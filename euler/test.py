import numpy as np

a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([1, 2, 3])
c = np.dot(a, b)
print(c)