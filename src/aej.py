# Advection with constant velocity vector in a double periodic domain

import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

from src.checkpointing import Checkpointing
from src.partial_derivatives import Partial
from src.time_marching_schemes import runge_kutta3, solve_ivp
from src.total_derivatives import antiderivative


# Model Parameters
n_days = 60
tau = 20 * 24 * 3600 # Seconds (Characteristic forcing timescale).
day_number = -1         # Day number (-1 for final day)
lapse_rate = 0.0065  # K/m

sahara_heating = 4.0   # Default 9.0
comp_cooling = -2.5    # Default -4.5
gulf_cooling = -0.8    # Default -3.0
convection = 0.0       # Default 0.0
cooling = 0.0          # Default 0.0

kappa_y = 200  # m² s⁻¹  # Default 1.0e2


# Define the temporal grid
tmin = 0.0              # Initial time
tmax = n_days * 86_400.0    # Final time
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
dt = 86_400


# Linear surface temperature profile for testing (Can be used without diffusion)
# def T_surface(Y):
#     """Define linear Surface temperature profile T(lat)"""
#     return 0.4 * Y + 298.15


def T_surface(y):
    """Define Half cosine meridional surface profile for Neumann boundary condition."""
    return 298.15 + 5.0 * (1.0 - np.cos(np.pi * y / 25.0))


def preprocessing():
    # Build grid
    Y, Z = np.meshgrid(y, z)

    # Apply lapse rate to meridional surface profile.
    T = T_surface(Y) - lapse_rate * Z

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


def equilibrium_temperature(grid):

    y = grid[0]
    z = grid[1]

    Teq = (298.15 - lapse_rate * z)

    return Teq


def source(grid, t):

    y = grid[0]
    z = grid[1]

    # Gaussian Saharan heating.
    # sahara = (
    #     (sahara_heating / tau_forcing)
    #     * np.exp(-((y - 20.0) / 8)**2)
    #     * np.exp(-((z - 2500.0) / 1600.0)**2)
    # )

    # Sigmoid Saharan heating
    sahara_y = (1.0 / (1.0 + np.exp(-(y - 15.0) / 2.5)))

    sahara = (
            (sahara_heating / tau)
            * sahara_y
            * np.exp(-((z - 2500.0) / 1600.0) ** 2)
    )

    # Upper-level compensating cooling
    upper_cooling = (
        (comp_cooling / tau)
        * np.exp(-((y - 20.0) / 5.0)**2)
        * np.exp(-((z - 4500.0) / 700.0)**2)
    )

    # Gulf of Guinea cooling
    gulf = (
        (gulf_cooling / tau)
        * np.exp(-((y - 5.0) / 4.0)**2)
        * np.exp(-z / 1200.0)
    )

    # Convective heating.
    convective_heating = (
            (convection / tau)
            * np.exp(-((y - 9.0) / 2.0) ** 2)
            * np.exp(-((z - 3000.0) / 1200.0) ** 2)
    )

    # Evaporative cooling beneath convective heating.
    evaporative_cooling = (
        (cooling / tau)
        * np.exp(-((y - 9.0) / 3.5)**2)
        * np.exp(-((z - 700.0) / 600.0)**2)
    )

    Q = sahara + upper_cooling + gulf + convective_heating + evaporative_cooling

    return Q

y_m = y * 111000.0

def diffusion(T):
    """
    Compute the meridional diffusion of temperature.

    Parameters
    ----------
    T : np.ndarray
        Temperature field with shape (nz, ny).

    Returns
    -------
    np.ndarray
        Meridional temperature diffusion term.
    """

    d2Tdy2 = Partial(T, y_m, z).neumann_dxx()

    return kappa_y * d2Tdy2


def rhs(t, T_flat):

    # Reshape 1-D state vector back into 2-D temperature field
    T = T_flat.reshape(nz, ny)

    # Temperature tendency
    dTdt = source(grid, t) - (T - Teq) / tau + diffusion(T)

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

T_final = T_all[day_number]


# Grid spacing in metres.
dy = y_m[1] - y_m[0]

dz = (zmax - zmin) / (nz - 1)

# Meridional temperature gradient
dTdy = Partial(T_final, y_m, z).biased_dx()

# Uncomment to test what happens the preprocessed T is used.
# dTdy = Partial(T, y_m, z).biased_dx()

# Thermal wind equation
dug_dz = -(g / (f * T_final)) * dTdy

ug_surface = -5.0   # m/s

ug = antiderivative(dug_dz, z, ug_surface)


# ==========================================================
# Plot final temperature field
# ==========================================================

plt.figure(figsize=(8, 5))

# Temperature contour levels at 5 K intervals
T_min = np.floor(T_final.min() / 5.0) * 5.0
T_max = np.ceil(T_final.max() / 5.0) * 5.0
levels = np.arange(T_min, T_max + 5.0, 2.0)

plt.contourf(
    grid[0],
    grid[1],
    T_final,
    levels=levels,
    cmap="plasma",
    extend="both",
)

plt.colorbar(
    label="Temperature (K)"
)

plt.xlabel("Latitude (degrees)")
plt.ylabel("Height (m)")
plt.title("Final Temperature Field")

plt.tight_layout()
plt.show()


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

vmin, vmax = -15, 15
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

plt.contourf(
    grid[0],
    grid[1],
    ug,
    levels=np.arange(vmin, vmax+1),          # optional: smoother contours
    cmap="RdBu_r",
    norm=norm,
    extend='both',
)

plt.colorbar(label="Geostrophic wind (m s$^{-1}$)", )
plt.xlabel("Latitude (degrees)")
plt.ylabel("Height (m)")
plt.title("African Easterly Jet")

plt.tight_layout()
plt.show()