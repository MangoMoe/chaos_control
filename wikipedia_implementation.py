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

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 40.0, 0.01)

states = odeint(f, state0, t)

fig = plt.figure()
ax = fig.gca(projection='3d')
# more stack overflow
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.plot(states[:, 0], states[:, 1], states[:, 2])

# how to plot surface from stack overflow, thanks
xx, yy = np.meshgrid(range(-30,30), range(-30,30))

# calculate corresponding z
# z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
z = np.ones((60,60)) * rho
ax.plot_surface(xx, yy, z, alpha=0.4)
plt.draw()
plt.show()