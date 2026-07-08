import matplotlib.pyplot as plt
import numpy as np


# Define the uniform, spatial grid for AEJ.
# x_min = -15.0
# x_max = 15.0
# nx = 60

y_min = 0.0
y_max = 25.0
ny = 50

z_min = 0.0
z_max = 5_000.0
nz = 1_000


# Define the temporal grid.
t_min = 0.0
t_max = 2.0
t_check = 0.05


# Define Reference profiles:
def sine_reference(grid):
    y = grid[0]
    z = grid[1]

    return np.sin(np.pi * y)


def logistic_reference(grid):
    y = grid[0]
    z = grid[1]

    return np.sin(np.pi * y)


def preprocessing():
    """Construct the grid and ICS."""
    y = np.linspace(y_min, y_max, ny+1)
    z = np.linspace(z_min, z_max, nz+1)

    grid = np.meshgrid(y, z)

    # Surface temperature latitude profile T(y).
    Y, Z = grid[0], grid[1]
    T_surface  = 0.4 * Y + 298.15

    # Surface temperature height profile T(z).
    gamma = 0.0065
    T = T_surface - gamma * Z

    return grid, T