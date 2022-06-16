import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    return 0, 1 - t + np.exp(-2*t), 0  # Derivatives
    #return 0, 1 - t - 4 * state[1], 0

def d_f(state, t):
    return 0, -1 + -2 * np.exp(-2*t), 0  # 2th Derivatives
    #return 0, -1, 0

state0 = np.array([20.0, 10.0, 5.0])
dt = 0.01
t = np.arange(0.0, 40.0, dt)

state = state0
delta = 0
states = []
for i in t:
    delta += np.array(d_f(state, i)) * dt
    state = state + delta * dt
    states.append(state)
states = np.array(states)

fig = plt.figure()
#ax = fig.gca(projection="3d")
#ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.plot(t, states[:, 1])

plt.draw()
plt.show()
