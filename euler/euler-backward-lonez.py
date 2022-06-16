# https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])  # Derivatives

def d_f(state, t):
    x, y, z = state  # Unpack the state vector
    return np.array([
        [sigma * - 1, sigma, 0],
        [rho - z, -1, x * - 1],
        [y, x, - beta]
    ])

state0 = np.array([20.0, 10.0, 5.0])
dt = 0.001
t = np.arange(0.0, 40.0, dt)

state = state0
states = []
delta = f(state0, 0)
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

for i in t:
    
    # 1
    # A = I - d_f(state, i) * dt
    # b = delta
    # delta = np.dot(np.linalg.inv(A), b)
    # state = state + delta  * dt
    
    # 2 forward
    # delta += np.dot(d_f(state, i), delta * dt) 

    # 3
    A =  I - d_f(state, i) * dt
    b = state + dt * f(state, 0) - np.dot(d_f(state, 0), state) * dt
    state = np.dot(np.linalg.inv(A), b)
    
    states.append(state)
states = np.array(states)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])

plt.draw()
plt.show()
