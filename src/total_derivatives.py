# %%
import numpy as np
from matplotlib import pyplot as plt


def biased_forward_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray:
    """
    Calculates the forward differencing approximation of the first-order
    derivative in uniform grids using biased backward difference approximation
    for the first step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the forward difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.empty_like(y)

    # Forward difference approximation on every other point.
    ddx[:-1] = h_inv * (y[1:] - y[:-1])

    # Backward difference approximation at last point.
    ddx[-1] = h_inv * (y[-1] - y[-2])

    return ddx


def periodic_forward_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Calculates the forward differencing approximation of the first-order
    derivative in uniform grids using periodic approximation for the end points.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the forward difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.empty_like(y)

    # Forward difference approximation on every other point.
    ddx[:-1] = h_inv * (y[1:] - y[:-1])

    # Forward difference approximation at last warping the first
    ddx[-1] = h_inv * (y[0] - y[-2])

    return ddx


def biased_backward_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Calculates the backward differencing approximation of the first-order
    derivative in uniform grids using biased forward difference approximation
    for the last step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the backward difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result
    ddx = np.empty_like(y)

    # Forward difference approximation at first point.
    ddx[0] = h_inv * (y[1] - y[0])

    # Backward difference at every point except the first.
    ddx[1:] = h_inv * (y[1:] - y[:-1])

    return ddx


def periodic_backward_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Calculates the backward differencing approximation of the first-order
    derivative in uniform grids using periodic backward differencing with
    the first step at the last step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the backward difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result
    ddx = np.empty_like(y)

    # Backward difference approximation at first point.
    ddx[:-1] = h_inv * (y[1:] - y[:-1])

    # Warping at last point.
    ddx[-1] = h_inv * (y[-1] - y[-2])

    return ddx


def biased_central_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Compute the central difference approximation of the first-order derivative
     in uniform grids using biased forward difference approximation for the first step
     and biased backward difference approximation for the last step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

     Returns - the central difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.zeros_like(y)

    # Forward difference approximation at first point.
    ddx[0] = h_inv * (y[1] - y[0])

    # Central difference at every point except the first and last.
    ddx[1:-1] = h_inv * 0.5 * (y[2:] - y[:-2])

    # Backward difference approximation at last point.
    ddx[-1] = h_inv * (y[-1] - y[-2])

    return ddx


def periodic_central_difference_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Compute the central difference approximation of the first-order derivative
     in uniform grids using periodic central difference at the boundaries.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

     Returns - the central difference approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.zeros_like(y)

    # Central difference approximation at first point with last point.
    ddx[0] = h_inv * 0.5 * (y[1] - y[-2])

    # Central difference at interior points.
    ddx[1:-1] = h_inv * 0.5 * (y[2:] - y[:-2])

    # Central difference approximation at last point with first point.
    ddx[-1] = h_inv * 0.5 * (y[0] - y[-3])

    return ddx


def biased_second_order_derivative_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Compute the second order derivative approximation of the function
    in uniform grids using biased forward difference approximation for the
    first step and biased backward difference approximation for the last step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the second order derivative approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.zeros_like(y)

    # Forward difference approximation at first point.
    ddx[0] = h_inv * h_inv * (y[0] - 2 * y[1] + y[2])

    # Interior central difference approximation in interior.
    ddx[1:-1] = h_inv * h_inv * (y[:-2] - 2 * y[1:-1] + y[2:])

    # Backward difference approximation at last point.
    ddx[-1] = h_inv * h_inv * (y[-3] - 2 * y[-2] + y[-1])

    return ddx


def periodic_second_order_derivative_approximation(
        function: np.ndarray,
        domain: np.ndarray
) \
        -> np.ndarray :
    """
    Compute the second order derivative approximation of the function
    in uniform grids using biased forward difference approximation for the
    first step and biased backward difference approximation for the last step.

    Parameters
        function - an array representing the function to be differentiated.
        domain - an array of the domain of the function.

    Returns - the second order derivative approximation of the function.
    """
    x = domain
    y = function
    h = x[1] - x[0]
    h_inv = 1 / h

    # Initialize the result.
    ddx = np.zeros_like(y)

    # Central difference approximation at first point with last point.
    ddx[0] = h_inv * h_inv * (y[-2] - 2 * y[0] + y[1])

    # Interior central difference approximation in interior.
    ddx[1:-1] = h_inv * h_inv * (y[:-2] - 2 * y[1:-1] + y[2:])

    # Backward difference approximation at last point.
    ddx[-1] = h_inv * h_inv * (y[-2] - 2 * y[-1] + y[1])

    return ddx


def antiderivative(
        dydx: np.ndarray,
        x: np.ndarray,
        y0: float
) \
        -> np.ndarray:
    """
    Integrate a vertical derivative using the trapezoidal rule.
    """

    y = np.zeros_like(dydx)

    y[0, :] = y0

    for k in range(1, len(x)):
        dx = x[k] - x[k - 1]

        y[k, :] = (
            y[k - 1, :]
            + 0.5
            * (y[k - 1, :] + y[k, :])
            * dx
        )

    return y


def main():
    x = np.linspace(-1, 1, 1_000)
    f1 = x
    f2 = np.sin(np.pi * x)

    fig, ax = plt.subplots()
    ax.plot(x, periodic_second_order_derivative_approximation(f2, x))
    ax.set_ylim(-10, 10)
    plt.show()



if __name__ == "__main__":
    main()