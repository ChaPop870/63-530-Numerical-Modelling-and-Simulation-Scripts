# %%
import matplotlib.pyplot as plt
import numpy as np

from src.time_marching_schemes import solve_ivp, runge_kutta3


# Parameters
n = 100
L = 1.0
c = 0.01
kappa = 0.1

h = L / n
h_inv = 1 / h


def transformation_dx(n):
    """Defines the n x n transformation matrix with centered first
    derivative with periodic BCs."""
    H = np.zeros((n, n))

    H += np.diag(np.ones(n - 1), k=1)
    H += np.diag(-1 * np.ones(n - 1), k=-1)

    H[0, -1] = -1
    H[-1, 0] = 1

    return H


def transformation_dxx(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix with centered second
    derivative with periodic BCs."""
    H = np.zeros((n, n))

    H += np.diag(-2 * np.ones(n))
    H += np.diag(np.ones(n - 1), k=1)
    H += np.diag(np.ones(n - 1), k=-1)

    H[0, -1] = 1
    H[-1, 0] = 1

    return H


# Spatial operators
dx = transformation_dx(n)
dxx = transformation_dxx(n)

# Semi-discrete operature du/dt = Au, where A
A = kappa * h_inv**2 * dxx - c * 0.5 * h_inv * dx


def adv_diff(t: float, u: np.ndarray) -> np.ndarray:
    return np.linalg.matmul(A, u)


# Periodic grid.
x = np.linspace(0, L, n, endpoint=False)

# Initial conditions.
t0 = 0.0
u0 = np.cos(2*np.pi * x)

tf = 2.0
dt = 1e-5

t, u = solve_ivp(
    adv_diff,
    t0=t0,
    u0=u0,
    tf=tf,
    dt=dt,
    solver=runge_kutta3
)


# Plotting
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, u[0], label=f"t={t[0]:.3f}")
ax.plot(x, u[len(t)//4], label=f"t={t[len(t)//4]:.3f}")
ax.plot(x, u[len(t)//2], label=f"t={t[len(t)//2]:.3f}")
ax.plot(x, u[-1], label=f"t={t[-1]:.3f}")

ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("Advection-Diffusion Equation")
ax.legend()

plt.show()