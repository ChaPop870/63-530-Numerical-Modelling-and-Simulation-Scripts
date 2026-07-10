import numpy as np
import matplotlib.pyplot as plt

from nums.pynums.fdms.fdm1 import fdm1_e2p
from nums.pynums.partials import partial
from nums.pynums.pdes.timemarching import *
from nums.pynums.pdes.checkpointing import *
from nums.pynums.pdes.postprocessing import *
from nums.pynums.iodata import *


# Define the spatial grid.
y_min = 0.0
y_max = 25.0
ny = 100

z_min = 0.0
z_max = 5_000.0
nz = 80


# Define the temporal grid.
t_min = 0.0
t_max = 20 * 86_400.0
t_check = 2 * 86_400.0


# Define the problem.
timescheme = RungeKutta3()
cnum = 0.20


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


def simulation(grid, u):
    pass


def postprocessing(grid, data):
    pass


def save(grid, data):
    pass


def source(grid,t):
    """Construct the source term: Saharan heating plus cooling in Gulf of Guinea"""
    Y = grid[0]
    Z = grid[1]

    # Sahara heating
    sahara = (3e-5 * np.exp(-((Y - 20) / 4)**2) * np.exp(-Z / 2500))

    # Gulf of Guinea cooling
    gulf = (-2e-5 * np.exp(-((Y - 5) / 4)**2) * np.exp(-Z / 2500))

    return sahara + gulf


def rhs(T,t):

    d2Tdz2 = np.zeros_like(T)
    d2Tdy2 = np.zeros_like(T)

    # vertical diffusion (second derivative in z)
    for j in range(ny):
        d2Tdz2[:, j] = fdm2_e121(T[:, j]) / dz**2

    # meridional diffusion (second derivative in y)
    for i in range(nx):
        d2Tdy2[i, :] = fdm2_e121(T[i, :]) / dy**2

    diffusion = K * (d2Tdz2 + d2Tdy2)

    # Correct source term
    S = source(grid, t)

    return diffusion + S