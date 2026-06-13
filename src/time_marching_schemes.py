from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


# Euler Method.
def euler(
        rhs: Callable[[float, float], float],
        tn: float,
        un: float,
        dt: float
) ->\
        tuple[float, float]:
    """Solves a single step of the Euler Method.

        Parameters:
        rhs - the function representing the RHS of the ODE to solve for.
        tn - the current time.
        un - the solution of the ODE.
        dt - the time step.

    Returns:
        t, u - the tuple containing the domain and the solution to the ODE
        in that domain.
    """
    return tn + dt, un + dt * rhs(tn, un)


# Third order Runge-Kutta Method
def runge_kutta3(
        rhs: Callable[[float, float], float],
        tn: float,
        un: float,
        dt: float
) ->\
        tuple[float, float]:
    """Solves a single step of the 3rd-order Runge-Kutta time-stepping
    method for solving an ODE.

    Parameters:
        rhs - the function representing the RHS of the ODE to solve for.
        tn - the current time.
        un - the solution of the ODE.
        dt - the time step.

    Returns:
        t, u - the tuple containing the domain and the solution to the ODE
        in that domain.
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
def solve_ivp(
        rhs: Callable[[float, float], float],
        t0: float,
        u0: float | np.ndarray,
        tf: float,
        dt: float,
        solver: Callable
) ->\
        tuple[np.ndarray, np.ndarray]:
    """Solves the IVP using the specified ODE solver.

    Parameters:
        rhs - the function representing the RHS of the ODE to solve for.
        t0 - the initial time.
        u0 - the initial value of the ODE.
        tf  - the final time.
        dt - the time step.

    Note: This function can solve both single IVPs and systems of ODEs.

    Returns:
        t, u - a tuple containing the domain and the solution of the ODE.
    """
    nt = int((tf - t0) / dt) + 1
    t = np.zeros(nt)

    # For systems of equations
    if np.isscalar(u0):
        u = np.zeros(nt)

    else:
        u = np.zeros((nt, len(u0)))

    # Initial conditions
    t[0] = t0
    u[0] = u0

    # Time integration loop
    for n in range(nt - 1):
        t[n + 1], u[n + 1] = solver(rhs, t[n], u[n], dt)

    return t, u


def lorenz(t: float, u: float) -> np.ndarray:
    """The Lorenz system."""
    x, y, z = u

    dxdt = -sigma*x + sigma*y
    dydt = -x*z + r*x - y
    dtdz = x*y - b*z

    return np.array([dxdt, dydt, dtdz])


# Parameters for Lorenz system.
sigma = 10
b = 8 / 3
r = 28


def main():
    # Initial conditions.
    u0 = np.array([0, 1, 0])
    t0 = 0

    tf = 60
    dt = 0.01

    t, u = solve_ivp(lorenz, t0, u0, tf, dt, solver=runge_kutta3)

    X = u[:, 0]
    Y = u[:, 1]
    Z = u[:, 2]

    # Plotting solutions of Lorenz equations.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    ax1.plot(t, X)
    ax1.set_ylabel('X(t)')
    ax1.set_xlim(t0, tf)

    ax2.plot(t, Y)
    ax2.set_ylabel('Y(t)')
    ax2.set_xlim(t0, tf)

    ax3.plot(t, Z)
    ax3.set_ylabel('Z(t)')
    ax3.set_xlabel('time / s')
    ax3.set_xlim(t0, tf)

    fig.suptitle("Lorenz Equations solution.", fontweight='bold')

    plt.show()


if __name__ == "__main__":
    main()