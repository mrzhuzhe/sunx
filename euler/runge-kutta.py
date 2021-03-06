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
    #return -1 * (x - y)*(1-x-y), x*(2+y), 0
    #return 4-2*y, 12 - 3*x*x, 0

state0 = np.array([20.0, 10.0, 5.0])
dt = 0.01
t = np.arange(0.0, 40, dt)

state = state0
#states2 = odeint(f, state0, t)

states = []
for i in t:
    K1 = np.array(f(state, t))  
    K2 = np.array(f(state + 0.5 * dt * K1, t + 0.5 * dt))
    K3 = np.array(f(state + 0.5 * dt * K2, t + 0.5 * dt))
    K4 = np.array(f(state + dt * K3, t + dt))
    state = state + 1/6 * (K1 + 2 * K2 + 2 * K3 + K4) * dt
    states.append(state)
states = np.array(states)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
#ax.plot(states2[:, 0], states2[:, 1], states2[:, 2])

plt.draw()
plt.show()
