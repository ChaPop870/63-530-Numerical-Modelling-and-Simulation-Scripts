import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

from src.checkpointing import Checkpointing
from src.partial_derivatives import Partial
from src.time_marching_schemes import runge_kutta3, solve_ivp
from src.total_derivatives import antiderivative


# Define constants.
g = 9.81                    # gravity (m/s²)
Omega = 7.292115e-5         # Earth's rotation (rad/s)
phi0 = 15.0                 # AEJ reference latitude (degrees)
T0 = 298.15                 # reference temperature (K)
f = 2 * Omega * np.sin(np.deg2rad(phi0))


# Model Parameters.
n_days = 60
tau = 20 * 24 * 3600 # Seconds (Characteristic forcing timescale).
tau_convection = 3 * 24 * 3600
day_number = -1      # Day number (-1 for final day)
lapse_rate = 0.0065  # K/m

# Forcing parameters.
sahara_heating = 7.0    # Default 7.0
comp_cooling = -2.5     # Default -2.5
gulf_cooling = -0.5     # Default -0.5
convection = 0#0.025        # Default 0.0
cooling = 0#0.017           # Default 0.0
sahara_start = 15.0     # Default 15.0
gulf_cooling_start = 5.0 # Default 5.0

kappa_y = 200.0  # m² s⁻¹  # Default 200.0


# Define the temporal grid
tmin = 0.0                  # Initial time
tmax = n_days * 86_400.0    # Final time
tcheck = 24 * 3600          # Time interval to checkpoint data


# Define the uniformly-spaced spatial grid.
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


# Linear surface temperature profile for testing (Can be used without diffusion).
# def T_surface(Y):
#     """Define linear surface temperature profile T(lat)"""
#     return 0.4 * Y + 298.15


def T_surface(y):
    """Define Half cosine meridional surface profile to satisfy Neumann boundary
    condition."""
    return T0 + 5.0 * (1.0 - np.cos(np.pi * y / 25.0))


def preprocessing():
    """Build the grid and starting temperature field using the surface temperature
    and the lapse rate."""
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
    """Defines the equilibrium temperature for Newtonian Cooling."""
    y = grid[0]
    z = grid[1]

    Teq = (298.15 - lapse_rate * z)

    return Teq


def source(grid, t):
    """Defines the sources and sinks and incorporates Newtonian Cooling."""
    y = grid[0]
    z = grid[1]

    # Gaussian Saharan heating.
    # sahara = (
    #     (sahara_heating / tau_forcing)
    #     * np.exp(-((y - 20.0) / 8)**2)
    #     * np.exp(-((z - 2500.0) / 1600.0)**2)
    # )

    # Sigmoid Saharan heating.
    sahara_y = (1.0 / (1.0 + np.exp(-(y - sahara_start) / 2.8)))

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
        * np.exp(-((y - gulf_cooling_start) / 4.0)**2)
        * np.exp(-z / 1200.0)
    )

    # Convective heating.
    convective_heating = (
            (convection / tau_convection)
            * np.exp(-((y - 11.0) / 4.0) ** 2)
            * np.exp(-((z - 3000.0) / 1200.0) ** 2)
    )

    # Evaporative cooling beneath convective heating.
    evaporative_cooling = (
        (cooling / tau_convection)
        * np.exp(-((y - 11.0) / 4.0)**2)
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


def rhs(t: np.ndarray, T_flat: np.ndarray) -> np.ndarray:
    """Defines the RHS of the temperature equation to be solved.

    ∂T/∂t = Q(y, z) − (T − T_eq)/τ_r + κ_y ∂²T/∂y²

    """
    # Reshape 1-D state vector back into 2-D temperature field
    T = T_flat.reshape(nz, ny)

    # Temperature tendency
    dTdt = source(grid, t) - (T - Teq) / tau + diffusion(T)

    # Flatten again for solve_ivp
    return dTdt.ravel()


def generate_monsoon():
    """Generate a low-level monsoon circulation for aesthetic purposes
    and does not affect the model."""
    U_monsoon = 5.0  # Maximum monsoon wind (m/s)
    y_monsoon = 10.0  # Latitude of maximum wind (degrees N)
    L_monsoon = 5.5  # Meridional width (degrees)
    H_monsoon = 400.0  # Height of maximum monsoon wind (m)

    # Meridional structure
    monsoon_y = np.exp(-((grid[0] - y_monsoon) / L_monsoon) ** 2)

    # Vertical structure:
    monsoon_z = (
            (grid[1] / H_monsoon)
            * np.exp(1.0 - grid[1] / H_monsoon)
    )

    u_monsoon = U_monsoon * monsoon_y * monsoon_z

    return u_monsoon


grid, T = preprocessing()

Teq = equilibrium_temperature(grid)

day_number = 60

def main():
    global grid, T, Teq, day_number
    # Main program.
    grid, T = preprocessing()

    Teq = equilibrium_temperature(grid)

    times, T_all, checkpoint = simulation(T)

    T_final = T_all[day_number]

    # Grid spacing in metres.
    # dy = y_m[1] - y_m[0]
    #
    # dz = (zmax - zmin) / (nz - 1)

    # Meridional temperature gradient
    dTdy = Partial(T_final, y_m, z).biased_dx()

    # Uncomment to test what happens if the preprocessed T is used.
    # dTdy = Partial(T, y_m, z).biased_dx()

    # Thermal wind equation
    dug_dz = -(g / (f * T_final)) * dTdy

    # ug_surface = 5.0 * np.tanh((20.0 - y) / 4.0)   # m/s
    #
    # ug = antiderivative(dug_dz, z, ug_surface)

    # ----------------------------------------------------------
    # Thermally driven wind
    # ----------------------------------------------------------

    ug_thermal = antiderivative(
        dydx=dug_dz,
        x=z,
        y0=0.0,
    )

    ug = ug_thermal + generate_monsoon()

    # ==========================================================
    # Plot temperature field
    # ==========================================================
    if day_number == -1:
        day_number = n_days

    fig, ax = plt.subplots(figsize=(8, 5))

    # Temperature contour levels at 5 K intervals.
    T_min = np.floor(T_final.min() / 5.0) * 5.0
    T_max = np.ceil(T_final.max() / 5.0) * 5.0
    levels = np.arange(T_min, T_max + 1, 2.0)

    temp_contours = ax.contourf(
        grid[0],
        grid[1],
        T_final,
        levels=levels,
        cmap="plasma",
        extend="both",
    )

    fig.colorbar(temp_contours, label="Temperature / K")

    ax.set_xlabel("Latitude / degrees")
    ax.set_ylabel("Height / m")
    ax.set_title(f"Final Temperature Field after {day_number} days.")

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Plot dT/dy
    # ==========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    limit = np.max(np.abs(dTdy))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    dTdy_plot = ax.pcolormesh(
        grid[0],
        grid[1],
        dTdy,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
    )

    fig.colorbar(dTdy_plot, label=r"$\partial T/\partial y$ (K m$^{-1}$)")

    ax.set_xlabel("Latitude / degrees")
    ax.set_ylabel("Height / m")
    ax.set_title(f"Meridional Temperature Gradient after {day_number} days.")

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Plot thermal wind shear
    # ==========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    limit = np.max(np.abs(dug_dz))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    dugdz_plot = ax.pcolormesh(
        grid[0],
        grid[1],
        dug_dz,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
    )

    fig.colorbar(dugdz_plot, label=r"$\partial u_g/\partial z$ (s$^{-1}$)")

    ax.set_xlabel("Latitude / degrees")
    ax.set_ylabel("Height / m")
    ax.set_title(f"Thermal Wind Shear after {day_number} days.")

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Plot geostrophic wind
    # ==========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    vmin, vmax = -12, 12
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    ug_plot = ax.contourf(
        grid[0],
        grid[1],
        ug,
        levels=np.arange(vmin, vmax + 1),  # optional: smoother contours
        cmap="RdBu_r",
        norm=norm,
        extend='both',
    )

    fig.colorbar(ug_plot, label="Geostrophic wind (m s$^{-1}$)")

    ax.set_xlabel("Latitude / degrees")
    ax.set_ylabel("Height / m")
    ax.set_title(f"African Easterly Jet after {day_number} days.")

    plt.tight_layout()
    plt.show()

    # Quick diffusion plot from AI for the sake of time.

    # # ==========================================================
    # # Run the model for a specified diffusion coefficient
    # # ==========================================================
    #
    # def run_model(kappa):
    #     """
    #     Run the AEJ model for a specified meridional diffusion coefficient.
    #
    #     Parameters
    #     ----------
    #     kappa : float
    #         Meridional temperature diffusivity (m^2 s^-1).
    #
    #     Returns
    #     -------
    #     T_final : np.ndarray
    #         Final temperature field.
    #     ug : np.ndarray
    #         Geostrophic wind field.
    #     dTdy : np.ndarray
    #         Meridional temperature gradient.
    #     dug_dz : np.ndarray
    #         Thermal wind shear.
    #     """
    #
    #     global kappa_y
    #
    #     # Set diffusion coefficient
    #     kappa_y = kappa
    #
    #     # Recreate initial condition
    #     grid, T = preprocessing()
    #
    #     # Equilibrium temperature
    #     global Teq
    #     Teq = equilibrium_temperature(grid)
    #
    #     # Run model
    #     times, T_all, checkpoint = simulation(T)
    #
    #     # Select final time
    #     T_final = T_all[day_number]
    #
    #     # Meridional temperature gradient
    #     dTdy = Partial(
    #         T_final,
    #         y_m,
    #         z
    #     ).biased_dx()
    #
    #     # Thermal wind shear
    #     dug_dz = -(g / (f * T_final)) * dTdy
    #
    #     # Thermally driven wind
    #     ug_thermal = antiderivative(
    #         dydx=dug_dz,
    #         x=z,
    #         y0=0.0
    #     )
    #
    #     # ------------------------------------------------------
    #     # Low-level monsoon circulation
    #     # ------------------------------------------------------
    #
    #     U_monsoon = 5.0
    #     y_monsoon = 10.0
    #     L_monsoon = 5.5
    #     H_monsoon = 400.0
    #
    #     monsoon_y = np.exp(
    #         -((grid[0] - y_monsoon) / L_monsoon)**2
    #     )
    #
    #     monsoon_z = (
    #         (grid[1] / H_monsoon)
    #         * np.exp(1.0 - grid[1] / H_monsoon)
    #     )
    #
    #     u_monsoon = U_monsoon * monsoon_y * monsoon_z
    #
    #     # Total geostrophic/diagnostic wind
    #     ug = ug_thermal + u_monsoon
    #
    #     return T_final, ug, dTdy, dug_dz
    #
    #
    #
    #
    # # ==========================================================
    # # Run both diffusion experiments
    # # ==========================================================
    #
    # T_diff, ug_diff, dTdy_diff, shear_diff = run_model(200.0)
    #
    # T_nodiff, ug_nodiff, dTdy_nodiff, shear_nodiff = run_model(0.0)
    #
    #
    # # ==========================================================
    # # Difference between diffusion and no-diffusion cases
    # # ==========================================================
    #
    # delta_T = T_diff - T_nodiff
    # delta_ug = ug_diff - ug_nodiff
    # delta_dTdy = dTdy_diff - dTdy_nodiff
    # delta_shear = shear_diff - shear_nodiff
    #
    # # ==========================================================
    # # Plot difference in geostrophic wind
    # # ==========================================================
    #
    # plt.figure(figsize=(8, 5))
    #
    # limit = np.max(np.abs(delta_ug))
    #
    # norm = TwoSlopeNorm(
    #     vmin=-limit,
    #     vcenter=0.0,
    #     vmax=limit
    # )
    #
    # levels = np.linspace(-0.005, 0.005, 21)
    #
    # plt.contourf(
    #     grid[0],
    #     grid[1],
    #     delta_ug,
    #     levels=levels,
    #     cmap="RdBu_r",
    #     norm=norm,
    #     extend="both"
    # )
    #
    # plt.colorbar(
    #     label=r"$\Delta u_g$ (m s$^{-1}$)"
    # )
    #
    # plt.xlabel("Latitude / degrees")
    # plt.ylabel("Height / m")
    #
    # plt.title(
    #     r"Effect of meridional diffusion on the AEJ: "
    #     r"$\kappa_y=200 - \kappa_y=0$"
    # )
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()