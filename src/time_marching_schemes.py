import matplotlib.pyplot as plt
import numpy as np


# Euler Method.
def euler(rhs, tn, yn, h):
    """Solves a single step of the Euler Method."""
    return yn + h * rhs(tn, yn)




# Parameters
sigma = 10
b = 8 / 3
r = 28

dt = 0.01
t_final = 60

# Initialize arrays.
t = np.arange(0, t_final + dt, dt)

X = np.zeros_like(t)
Y = np.zeros_like(t)
Z = np.zeros_like(t)

# Initial conditions
X[0] = 0
Y[0] = 1
Z[0] = 0


def lorenz(x, y, z):
    dxdt = -sigma*x + sigma*y
    dydt = -x*z + r*x - y
    dtdz = x*y - b*z

    return dxdt, dydt, dtdz


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(t, X)
ax2.plot(t, Y)
ax3.plot(t, Z)

plt.show()