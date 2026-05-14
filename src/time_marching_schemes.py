import matplotlib.pyplot as plt
import numpy as np


# Euler Method.
def euler(rhs, tn, un, dt):
    """Solves a single step of the Euler Method."""
    return tn + dt, un + dt * rhs(tn, un)


# Third order Runge-Kutta Method
def runge_kutta3(rhs, tn, un, dt):
    """Solves a single step of the 3rd-order Runge-Kutta time-stepping
    method for solving an ODE.

    Parameters:
        rhs - The function representing the RHS of the ODE to solve for.
        tn - the current time.
        un - the solution of the ODE.
        dt - the time step.
    """

    # First slope estimate.
    k1 = rhs(tn, un)

    # Mid-point estimate.
    k2 = rhs(tn + 0.5*dt,
             un + 0.5*dt*k1)

    # Third slope estimate.
    k3 = rhs(tn + dt,
             un + dt * (2*k2 - k1))

    # Final weighted average.
    u = un + dt * (k1 + 4*k2 + k3) / 6

    return tn + dt, u


# Solve the initial value problem.
def solve_ivp(rhs, t0, u0, tf, dt, solver):
    nt = int((tf - t0) / dt) + 1
    t = np.zeros(nt)
    u = np.zeros(nt)

    # Initial conditions
    t[0] = t0
    u[0] = u0

    # Time integration loop
    for n in range(nt - 1):
        t[n + 1], u[n + 1] = solver(rhs, t[n], u[n], dt)

    return t, u


# Euler Method.
def euler(rhs, tn, yn, h):
    """Solves a single step of the Euler Method."""
    return tn + dt, un + dt * rhs(tn, un)


# Third order Runge-Kutta Method
def runge_kutta3(rhs, tn, un, dt):
    """Solves a single step of the 3rd-order Runge-Kutta time-stepping
    method for solving an ODE.

    Parameters:
        rhs - The function representing the RHS of the ODE to solve for.
        tn - the current time.
        un - the solution of the ODE.
        dt - the time step.
    """

    # First slope estimate.
    k1 = rhs(tn, un)

    # Mid-point estimate.
    k2 = rhs(tn + 0.5*dt, un + 0.5*dt*k1)

    # Third slope estimate.
    k3 = rhs(tn + dt, un + dt * (2*k2 - k1))

    # Final weighted average.
    u = un + dt * (k1 + 4*k2 + k3) / 6

    return tn + dt, u


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