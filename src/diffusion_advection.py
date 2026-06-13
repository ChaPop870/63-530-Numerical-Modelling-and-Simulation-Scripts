# %%
import matplotlib.pyplot as plt
import numpy as np

from src.time_marching_schemes import *


n = 100
kappa = 1.0
c = 1.0
h = 100
h_inv = 1 / h


def transformation_dx(n):
    """Defines the n x n transformation matrix necessary to solve the
    first-order partial derivative wrt x using the periodic boundary
    condition."""
    H = np.zeros((n, n))
    H += np.diag(np.ones(n - 1), k=1)
    H += np.diag(-1 * np.ones(n - 1), k=-1)
    H[0, -1] = -1
    H[-1, 0] = 1
    return H


def transformation_dxx(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix necessary to solve the
    second order partial derivative wrt x using the periodic boundary
    condition."""
    H = np.zeros((n, n))
    H += np.diag(-2 * np.ones(n))
    H += np.diag(np.ones(n - 1), k=1)
    H += np.diag(np.ones(n - 1), k=-1)
    H[0, -1] = 1
    H[-1, 0] = 1
    return H


dx = transformation_dx(n)
dxx = transformation_dxx(n)

A = kappa * h_inv**2 * dxx - c * 0.5 * h_inv * dx


def adv_diff(t: float, u: np.ndarray) -> np.ndarray:
    return np.linalg.matmul(A, u)


t0 = 0
x = np.linspace(0, 1, n)
u0 = np.sin(2*np.pi*x)
tf = 60
dt = h

t, u = solve_ivp(adv_diff, t0=t0, u0=u0, tf=tf, dt=dt, solver=runge_kutta3)


fig, ax = plt.subplots()
ax.plot(t, u)
plt.show()