import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    #return 0, 1 - t + np.exp(-2*t), 0  # Derivatives
    #return 0, 1 - t - 4 * state[1], 0
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = np.array([20.0, 10.0, 5.0])
dt = 0.01
t = np.arange(0.0, 40.0, dt)

state = state0
states2 = odeint(f, state0, t)

states = []
for i in t:
    f_t_0 = np.array(f(state, i))
    _estimate_state = state + 0.5 * f_t_0 * dt
    f_t_1 = np.array(f(_estimate_state, i))
    state = state + 0.5 * (f_t_0 + f_t_1) * dt
    states.append(state)
states = np.array(states)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.plot(states2[:, 0], states2[:, 1], states2[:, 2])
#plt.plot(t, states[:, 1])

plt.draw()
plt.show()
