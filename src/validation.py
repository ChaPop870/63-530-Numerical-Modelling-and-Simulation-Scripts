# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

from src.partial_derivatives import Partial
from src.time_marching_schemes import runge_kutta3, solve_ivp
from src.total_derivatives import antiderivative


mpl.rcParams['axes.labelsize'] = 16      # fontsize for x/y labels
mpl.rcParams['xtick.labelsize'] = 14     # fontsize for x‑tick labels
mpl.rcParams['ytick.labelsize'] = 14     # fontsize for y‑tick labels
mpl.rcParams['axes.titlesize'] = 18      # fontsize for axes titles


# Test 1: Testing the Runge-Kutta method and the solver.

def u_rhs(t, u):
    """Define du/dt = - u for testing."""
    return - u


def u_analytic(t):
    """Define analytic solution of du/dt = - u for testing."""
    return np.exp(-t)


# Initial conditions for testing u.
t0 = 0
u0 = 1
tf = 5
dt = 0.01

t, u_numerical = solve_ivp(u_rhs, t0, u0, tf, dt, solver=runge_kutta3)
u_analytic = u_analytic(t)


# Plotting.
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

ax.plot(t, u_numerical, label="Numerical solution", linestyle='dashed')
ax.plot(t, u_analytic, label="Analytic solution", alpha=0.5)
ax.set_xlabel("t")
ax.set_ylabel("u")
ax.set_title(r"Numerical solution vs Analytic solution of $\dfrac{\mathrm{d}u}{\mathrm{d}t} = -u$", y=1.02)
ax.legend()

plt.show()


# Test 2: Testing the partial differentiation.

def T_analytic(y, z):
    """
    Analytical temperature field:
        T(y,z) = sin(y) cos(z)
    """
    return np.sin(y) * np.cos(z)


def dTdy_analytic(y, z):
    """
    Analytical derivative:
        dT/dy = cos(y) cos(z)
    """
    return np.cos(y) * np.cos(z)

y = np.linspace(0, 2*np.pi, 101)
z = np.linspace(0, 2*np.pi, 101)
Y, Z = np.meshgrid(y, z)


# Define temperature field.
T = T_analytic(Y, Z)

# Compute partial derivatives.
dTdy_numerical = Partial(T, y, z).biased_dx()
dTdy_exact = dTdy_analytic(Y, Z)

# Compute errors.
error = dTdy_numerical - dTdy_exact

max_error = np.max(np.abs(error))
rmse = root_mean_squared_error(dTdy_exact, dTdy_numerical)

print("Maximum absolute error:", max_error)
print("RMSE:", rmse)


# Plotting temperature field.
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

ax1, ax2, ax3, ax4 = axes.flatten()

# ax1: Temperature field plot.
pcm1 = ax1.pcolormesh(
    Y,
    Z,
    T,
    shading="auto",
    cmap="viridis"
)

fig.colorbar(
    pcm1,
    ax=ax1,
    label=r"$T(y,z)$"
)

ax1.set_xlabel(r"$y$")
ax1.set_ylabel(r"$z$")
ax1.set_title(r"Analytical temperature field $T(y,z)=\sin(y)\cos(z)$")


# ax2: Plot Analytical derivative.
pcm2 = ax2.pcolormesh(
    Y,
    Z,
    dTdy_exact,
    shading="auto",
    cmap="coolwarm"
)

fig.colorbar(
    pcm2,
    ax=ax2,
    label=r"Analytical $\partial T/\partial y$"
)

ax2.set_xlabel(r"$y$")
ax2.set_ylabel(r"$z$")
ax2.set_title(
    r"Analytical derivative $\partial T/\partial y$"
)

# ax3: Plot Numerical derivative.

pcm3 = ax3.pcolormesh(
    Y,
    Z,
    dTdy_numerical,
    shading="auto",
    cmap="coolwarm"
)

fig.colorbar(
    pcm3,
    ax=ax3,
    label=r"Numerical $\partial T/\partial y$"
)

ax3.set_xlabel(r"$y$")
ax3.set_ylabel(r"$z$")
ax3.set_title(r"Numerical derivative using `Partial.biased_dx()`")


# ax4: Plot Error.
pcm4 = ax4.pcolormesh(
    Y,
    Z,
    error,
    shading="auto",
    cmap="RdBu_r"
)

fig.colorbar(
    pcm4,
    ax=ax4,
    label=r"Numerical - Analytical"
)

ax4.set_xlabel(r"$y$")
ax4.set_ylabel(r"$z$")
ax4.set_title(
    r"Derivative error: "
    r"$\partial T/\partial y|_{\mathrm{num}}"
    r"-\partial T/\partial y|_{\mathrm{exact}}$"
)


# Test 3: Testing second derivative of partial derivative
def d2udx2_analytic(x):
    """Analytical derivative of
        f(x) = cos(2𝝅x)
    """
    return -(2 * np.pi)**2 * np.cos(2 * np.pi * x)


nx = 101
x = np.linspace(0.0, 1.0, nx)

f = np.cos(2 * np.pi * x)

# Numerical second derivative using Partial
d2f_numerical = Partial(f[np.newaxis, :], x).neumann_dxx()[0]

# Analytical second derivative
d2f_exact = d2udx2_analytic(x)

# Error
error = d2f_numerical - d2f_exact

print(f"Maximum absolute error: {np.max(np.abs(error)):.6e}")
print(f"RMSE: {np.sqrt(np.mean(error**2)):.6e}")


# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Original function
ax1.plot(x, f, label="f(x)")
ax1.set_ylabel("f(x)")
ax1.set_title("Test function")
ax1.legend()


# Second derivative
ax2.plot(
    x,
    d2f_exact,
    label="Analytical $f''(x)$"
)

ax2.plot(
    x,
    d2f_numerical,
    "--",
    label="Numerical $f''(x)$"
)

ax2.set_ylabel("$f''(x)$")
ax2.set_title("Second derivative")
ax2.legend()


# Error
ax3.plot(
    x,
    error,
    label="Numerical - Analytical"
)

ax3.axhline(0, linestyle="--")
ax3.set_xlabel("x")
ax3.set_ylabel("Error")
ax3.set_title("Second derivative error")
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, 1)


plt.tight_layout()
plt.show()


# Test 4: Vertical integration.

def dydz_analytic(z):
    """
    Analytical derivative:
        dy/dx = sin(z)
    """
    return np.sin(z)


def y_analytic(z):
    """
    Analytical integral satisfying y(0) = 0:
        y(x) = 1 - cos(z)
    """
    return 1.0 - np.cos(z)


z = np.linspace(0.0, 2.0 * np.pi, 101)
dydz = dydz_analytic(z)
dydz_2d = dydz[:, np.newaxis]

y_numerical = antiderivative(
    dydx=dydz_2d,
    x=z,
    y0=0.0
)

# Convert back to a one-dimensional array
y_numerical = y_numerical[:, 0]

# Analytical solution
y_exact = y_analytic(z)

# Calculate error
error = y_numerical - y_exact

max_error = np.max(np.abs(error))
rmse = np.sqrt(np.mean(error**2))

print("Maximum absolute error:", max_error)
print("RMSE:", rmse)


# ==========================================================
# Plot numerical vs analytical solution
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

ax1, ax2 = axes

# ----------------------------------------------------------
# Left: Numerical and analytical solutions
# ----------------------------------------------------------

ax1.plot(
    z,
    y_exact,
    label="Analytical solution",
    linewidth=2
)

ax1.plot(
    z,
    y_numerical,
    "--",
    label="Numerical solution",
    linewidth=2
)

ax1.set_xlabel(r"$z$")
ax1.set_ylabel(r"$y$")
ax1.set_title(r"Vertical integration: $dy/dz = \sin(z)$")
ax1.legend()
ax1.grid(alpha=0.3)

# ----------------------------------------------------------
# Right: Error
# ----------------------------------------------------------

ax2.plot(
    z,
    error,
    linewidth=2
)

ax2.axhline(
    0.0,
    linestyle="--",
    linewidth=1
)

ax2.set_xlabel(r"$z$")
ax2.set_ylabel(r"Numerical - Analytical")
ax2.set_title("Integration error")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()