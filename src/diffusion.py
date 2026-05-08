# %%
import matplotlib.pyplot as plt
import numpy as np


def bcs(lower: float, upper: float, function: np.ndarray) -> np.ndarray:
    """
    Boundary conditions vector given the values of the lower boundary, upper boundary and the
    function.
    """
    bcs_vector = np.zeros_like(function)
    bcs_vector[0] = lower
    bcs_vector[-1] = upper
    return bcs_vector


def diffusion_matrix(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix necessary to solve the diffusion problem."""
    H = np.zeros((n, n))
    H += np.diag(-1 * np.ones(n - 1), k=1)
    H += np.diag(2 * np.ones(n))
    H += np.diag(-1 * np.ones(n - 1), k=-1)

    # Von Neumann boundary condition at bottom.
    H[0, :] = 0
    H[0, :3] = np.array([1.5, -2.0, 0.5])

    # Dirichlet boundary condition at top.
    H[-1, :] = 0
    H[-1, -1] = 1.0
    return H


def test_functions(source: np.ndarray, analytic_sol: np.ndarray, boundary_conditions: np.ndarray) -> np.ndarray:
    """Test functions"""
    return source, analytic_sol, boundary_conditions


# Define domain and spacing.
n = 101
x = np.linspace(0, 1, n)
dx = x[1] - x[0]

# Source
s1 = 10 * np.sin(np.pi * x)
# s2 = (0.5 * np.pi)**2 * np.cos(0.5 * np.pi * x)
s3 = 6 * x
s4 = - np.exp(x)
# u_analytic = - np.cos(0.5 * np.pi * x)

# Correct Analytic solution.
u_analytic = - (10 / np.pi**2) * np.sin(np.pi * x)
# u_analytic = 1 - x**3
# u_analytic = np.exp(x) - x

# H_inv = np.linalg.inv(H)

# Boundary Conditions
# lower = 0.5 * dx * (3 * u_analytic[2] - 4 * u_analytic[1] + u_analytic[0])
lower = dx * 10 / np.pi
# lower = 0
upper = 0
bcs = bcs(lower, upper, s1)

H = diffusion_matrix(n)
H_inv = np.linalg.inv(H)

u = dx**2 * np.linalg.inv(H) @ s

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x, s)
ax2.plot(x, u)
plt.show()