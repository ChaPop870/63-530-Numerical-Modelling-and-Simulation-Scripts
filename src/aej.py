# Advection with constant velocity vector in a double periodic domain

import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

from src.checkpointing import Checkpointing
from src.partial_derivatives import Partial
from src.time_marching_schemes import runge_kutta3, solve_ivp


# Define the temporal grid
tmin = 0.0              # Initial time
tmax = 20 * 86_400.0    # Final time
tcheck = 24 * 3600      # Time interval to checkpoint data


# Define the spatial grid, uniformly spaced
ymin = 0.0
ymax = 25.0
ny = 101
y = np.linspace(ymin, ymax, ny)

zmin = 0.0
zmax = 5_000.0
nz = 101
z = np.linspace(zmin, zmax, nz)


# Define the problem.
dt = 300.0         # 5 minutes


def preprocessing():
    # Build grid
    Y, Z = np.meshgrid(y, z)

    # Surface temperature profile T(lat)
    T_surface = 0.4 * Y + 298.15

    # Apply lapse rate
    lapse_rate = 0.0065  # K/m
    T = T_surface - lapse_rate * Z

    return (Y, Z), T


def simulation(T):

    checkpoint = Checkpointing(delta_t=tcheck, t0=tmin)

    # Flatten initial temperature field
    T0 = T.ravel()

    # Solve the IVP
    t, T_flat = solve_ivp(
        rhs=rhs,
        t0=tmin,
        u0=T0,
        tf=tmax,
        dt=dt,
        solver=runge_kutta3,
        checkpoint=checkpoint
    )

    # Reshape each time step back to (nz, ny)
    T = T_flat.reshape(len(t), nz, ny)

    return t, T, checkpoint


tau = 20 * 24 * 3600


def equilibrium_temperature(grid):

    y = grid[0]
    z = grid[1]

    lapse = 0.0065  # K/m

    Teq = (
        298.15
        - lapse * z
    )

    return Teq


def source(grid, t):

    y = grid[0]
    z = grid[1]

    # Characteristic forcing timescale
    tau_forcing = 20 * 24 * 3600.0

    # Sahara heating
    sahara = (
        (15.0 / tau_forcing)
        * np.exp(-((y - 20.0) / 5.5)**2)
        * np.exp(-((z - 2500.0) / 800.0)**2)
    )

    # Upper-level compensating cooling
    upper_cooling = (
        (-8.0 / tau_forcing)
        * np.exp(-((y - 20.0) / 5.0)**2)
        * np.exp(-((z - 4500.0) / 700.0)**2)
    )

    # Gulf of Guinea cooling
    gulf = (
        (-5.0 / tau_forcing)
        * np.exp(-((y - 5.0) / 4.0)**2)
        * np.exp(-z / 1200.0)
    )

    Q = sahara + upper_cooling + gulf

    return Q


def rhs(t, T_flat):

    # Reshape 1-D state vector back into 2-D temperature field
    T = T_flat.reshape(nz, ny)

    # Temperature tendency
    dTdt = source(grid, t) - (T - Teq) / tau

    # Flatten again for solve_ivp
    return dTdt.ravel()


# Define constants.
g = 9.81                    # gravity (m/s²)
Omega = 7.292115e-5         # Earth's rotation (rad/s)
phi0 = 15.0                 # representative latitude (degrees)
T0 = 300.0                  # reference temperature (K)
f = 2 * Omega * np.sin(np.deg2rad(phi0))

grid, T = preprocessing()

Teq = equilibrium_temperature(grid)

times, T_all, checkpoint = simulation(T)

T_final = T_all[-1]


# Grid spacing in metres.
y_m = y * 111000.0
dy = y_m[1] - y_m[0]

dz = (zmax - zmin) / (nz - 1)

# Meridional temperature gradient
dTdy = Partial(T_final, y_m, z).biased_dx()

# Thermal wind equation
dug_dz = -(g / (f * T_final)) * dTdy

# Integrate downward from top boundary
ug = np.zeros_like(T_final)

# Surface reference wind
ug_surface = -5.0   # m/s
ug[0,:] = ug_surface

for k in range(1, nz):

    ug[k,:] = (
        ug[k-1,:]
        +0.5*(dug_dz[k-1,:] + dug_dz[k,:])*dz
    )


# ==========================================================
# Plot dT/dy
# ==========================================================

plt.figure(figsize=(8,5))

limit = np.max(np.abs(dTdy))
norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

plt.pcolormesh(
    grid[0],
    grid[1],
    dTdy,
    shading="auto",
    cmap="coolwarm",
    norm=norm,
)

plt.colorbar(label=r"$\partial T/\partial y$ (K m$^{-1}$)")
plt.xlabel("Latitude (degrees)")
plt.ylabel("Height (m)")
plt.title("Meridional Temperature Gradient")

plt.tight_layout()
plt.show()

# ==========================================================
# Plot thermal wind shear
# ==========================================================

plt.figure(figsize=(8,5))

limit = np.max(np.abs(dug_dz))
norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

plt.pcolormesh(
    grid[0],
    grid[1],
    dug_dz,
    shading="auto",
    cmap="RdBu_r",
    norm=norm,
)

plt.colorbar(label=r"$\partial u_g/\partial z$ (s$^{-1}$)")
plt.xlabel("Latitude (degrees)")
plt.ylabel("Height (m)")
plt.title("Thermal Wind Shear")

plt.tight_layout()
plt.show()

# ==========================================================
# Plot geostrophic wind
# ==========================================================

plt.figure(figsize=(8,5))

limit = np.max(np.abs(ug))
norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

plt.pcolormesh(
    grid[0],
    grid[1],
    ug,
    shading="auto",
    cmap="RdBu_r",
    norm=norm,
)

plt.colorbar(label="Geostrophic wind (m s$^{-1}$)")
plt.xlabel("Latitude (degrees)")
plt.ylabel("Height (m)")
plt.title("African Easterly Jet")

plt.tight_layout()
plt.show()