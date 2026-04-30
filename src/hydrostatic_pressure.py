# %%
import matplotlib.pyplot as plt
import numpy as np


g = 9.81
Rd = 287.1
gamma = 6.5e-3

z = np.linspace(0, 100_000, 100_001)
theta_0 = 1
theta_profile  = 300.0 + gamma * z


def exner(theta: float | np.ndarray) -> float | np.ndarray:
    exner_profile = - g / (Rd * theta)
    return exner_profile


def transformation_matrix(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix necessary to solve the problem."""
    H = np.zeros((n, n))
    H += np.diag(0.5 * np.ones(n - 1), k=1)
    H += np.diag(-0.5 * np.ones(n - 1), k=-1)
    H[0, :2] = np.array([-1, 1])
    H[-1, -2:] = np.array([-1, 1])
    return H


y = exner(theta_profile)

fig, ax = plt.subplots()
ax.plot(y, z)
plt.show()