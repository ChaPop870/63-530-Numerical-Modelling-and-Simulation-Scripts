
kappa = 1.0
c = 1.0


def transformation_dx(n):
    """Defines the n x n transformation matrix necessary to solve the
    first-order partial derivative wrt x using the periodic boundary
    condition."""
    H = np.zeros((n, n))
    H += np.diag(np.ones(n - 1), k=1)
    H += np.diag(-1 * np.ones(n - 1), k=-1)
    H[0, -1] = -1
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


print(transformation_dxx(5))