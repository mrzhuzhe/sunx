import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = np.array([20.0, 10.0, 5.0])
dt = 0.01
t = np.arange(0.0, 40.0, dt)

states2 = odeint(f, state0, t)
state = state0
states = []
for i in t:
    delta = np.array(f(state, 0))
    state = state + delta * dt
    states.append(state)
states = np.array(states)

fig = plt.figure()
ax = fig.gca(projection="3d")
#ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.plot(states2[:, 0], states2[:, 1], states2[:, 2])

plt.draw()
plt.show()
