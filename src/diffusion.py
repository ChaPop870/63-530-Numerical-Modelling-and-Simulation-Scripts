# %%
import matplotlib.pyplot as plt
import numpy as np


def bcs(lower: float, upper: float, domain: np.ndarray) -> np.ndarray:
    """
    Boundary conditions vector given the values of the lower boundary, upper boundary and the
    function.
    """
    bcs_vector = np.zeros_like(domain)
    bcs_vector[0] = lower
    bcs_vector[-1] = upper
    return bcs_vector


def diffusion_matrix(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix necessary to solve the diffusion problem using
    the Neumann boundary condition at the lower boundary and the Dirichlet boundary condition
    at the upper boundary."""

    # Initialize the n x n null matrix.
    H = np.zeros((n, n))

    # Interior stencil.
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


def solve_system(transformation_matrix: np.ndarray, source: np.ndarray, boundary_conditions: np.ndarray) -> np.ndarray:
    rhs = - dx**2 * source + boundary_conditions
    return np.linalg.solve(transformation_matrix, rhs)


# Define domain and spacing.
n = 101
x = np.linspace(0, 1, n)
dx = x[1] - x[0]

# Define source functions.
s1 = 10 * np.sin(np.pi * x)
s2 = (0.5 * np.pi)**2 * np.cos(0.5 * np.pi * x)
s3 = 6 * x
s4 = np.exp(x)

# Define correct Analytic solution.
u1_analytic = - (10 / np.pi**2) * np.sin(np.pi * x)
u2_analytic = - np.cos(0.5 * np.pi * x)
u3_analytic = 1 - x**3
u4_analytic = np.exp(x) - x

# Define boundary conditions.
lower1, upper1 = (dx * 10 / np.pi, 0)
bcs1 = bcs(lower1, upper1, x)

lower2, upper2 = (0, 0)
bcs2 = bcs(lower2, upper2, x)

lower3, upper3 = (0, 1)
bcs3 = bcs(lower3, upper3, x)

lower4, upper4 = (0, 0)
bcs4 = bcs(lower4, upper4, x)

# Define the transformation matrix specific to the diffusion problem.
H = diffusion_matrix(n)

rhs = - dx**2 * s1 + bcs

u = np.linalg.solve(H, rhs)

# u = H_inv @ (s1 * -dx**2 + bcs)
# y = 10 / ((np.pi)**2) * np.sin(np.pi * x)
# y = x**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("Source term s(x)")
ax1.plot(x, s1)
ax2.set_title("Solution")
ax2.plot(x, u, label="Numerical")
ax2.plot(x, u_analytic, "--", label="Analytic")

ax2.legend()

plt.show()