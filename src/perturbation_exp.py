from aej import *


# Add a localized temperature perturbation.
perturbation_amplitude = 2.0    # K
perturbation_y = 15.0           # degrees N
perturbation_z = 2500.0         # m
perturbation_Ly = 3.0           # degrees
perturbation_Lz = 1000.0        # m


def temperature_perturbation(grid):
    """Adds a localized temperature perturbation."""
    y = grid[0]
    z = grid[1]

    perturbation = (
        perturbation_amplitude
        * np.exp(
            -((y - perturbation_y) / perturbation_Ly)**2
        )
        * np.exp(
            -((z - perturbation_z) / perturbation_Lz)**2
        )
    )

    return perturbation


def main():
    # 1. Basic-state simulation.
    n_days_basic = 60

    tmin = 0.0
    tmax = n_days_basic * 86400.0

    grid, T_initial = preprocessing()
    Teq = equilibrium_temperature(grid)

    _, T_all_basic, _ = simulation(T_initial)

    # Final basic-state temperature
    T_basic = T_all_basic[-1]


    # 2. Diagnose basic-state of AEJ.
    dTdy_basic = Partial(
        T_basic,
        y_m,
        z
    ).biased_dx()

    dug_dz_basic = (
        -(g / (f * T_basic))
        * dTdy_basic
    )

    ug_thermal_basic = antiderivative(
        dydx=dug_dz_basic,
        x=z,
        y0=0.0
    )

    ug_basic = ug_thermal_basic + generate_monsoon()

    T_perturbed = T_basic + temperature_perturbation(grid)

    n_days_perturbation = 20

    tmax = n_days_perturbation * 86400.0

    times, T_perturbed_all, _ = simulation(T_perturbed)


    # 3. Diagnose the AEJ throughout perturbation experiment.
    ug_all = np.zeros_like(T_perturbed_all)

    for n in range(len(times)):

        T_current = T_perturbed_all[n]

        # Meridional temperature gradient
        dTdy = Partial(
            T_current,
            y_m,
            z
        ).biased_dx()

        # Thermal wind shear
        dug_dz = (
            -(g / (f * T_current))
            * dTdy
        )

        # Vertically integrate thermal-wind shear
        ug_thermal = antiderivative(
            dydx=dug_dz,
            x=z,
            y0=0.0
        )

        # Add prescribed monsoon circulation
        ug_all[n] = ug_thermal + generate_monsoon()


    delta_ug = ug_all - ug_basic

    T_anomaly = T_perturbed_all - T_basic

    days_to_plot = [0, 1, 3, 5, 10, 20]


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    limit = np.max(np.abs(T_anomaly))

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit
    )

    for ax, day in zip(axes, days_to_plot):

        i = np.argmin(
            np.abs(times / 86400.0 - day)
        )

        cf = ax.contourf(
            grid[0],
            grid[1],
            T_anomaly[i],
            levels=21,
            cmap="RdBu_r",
            norm=norm,
            extend="both"
        )

        ax.set_title(f"Day {day}")

        ax.set_xlabel("Latitude / degrees")
        ax.set_ylabel("Height / m")

    fig.colorbar(
        cf,
        ax=axes,
        label=r"$T' = T-T_{\mathrm{basic}}$ / K"
    )

    fig.suptitle(
        "Evolution of Temperature Perturbation from Thermal-Wind Equilibrium",
        fontsize=14
    )

    plt.show()


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    limit = np.max(np.abs(delta_ug))

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit
    )

    for ax, day in zip(axes, days_to_plot):

        i = np.argmin(
            np.abs(times / 86400.0 - day)
        )

        cf = ax.contourf(
            grid[0],
            grid[1],
            delta_ug[i],
            levels=21,
            cmap="RdBu_r",
            norm=norm,
            extend="both"
        )

        ax.set_title(f"Day {day}")

        ax.set_xlabel("Latitude / degrees")
        ax.set_ylabel("Height / m")

    fig.colorbar(
        cf,
        ax=axes,
        label=r"$\Delta u_g$ / m s$^{-1}$"
    )

    fig.suptitle(
        "Evolution of AEJ Perturbation Following a Temperature Disturbance",
        fontsize=14
    )

    plt.show()


if __name__ == '__main__':
    main()