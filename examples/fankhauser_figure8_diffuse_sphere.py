import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import lumos.calculator
import lumos.constants
import lumos.conversions


# ------------------------------------------------------------
# Fankhauser et al. Figure 8: Diffuse Sphere Model
# ------------------------------------------------------------

sat_height = 550 * 1000          # 550 km
sun_azimuth = 90                 # degrees
sun_altitudes = [-27, -21, -15, -9, -3]

# Best-fit albedo-area product from the paper
rho_area_product = 0.65          # rho * A


# Observer sky grid:
# altitude 0° = horizon, altitude 90° = zenith
sat_altitudes, sat_azimuths = np.meshgrid(
    np.linspace(0, 90, 45),
    np.linspace(0, 360, 90)
)


def diffuse_sphere_intensity(
    sat_altitudes,
    sat_azimuths,
    sun_altitude
):
    """
    Calculate diffuse-sphere intensity using Equation 30
    from Fankhauser et al. (2023).
    """

    # Convert observer-frame geometry into Lumos brightness coordinates
    obs_x, obs_y, obs_z, angle_past_terminator = (
        lumos.calculator.get_brightness_coords(
            sat_altitudes,
            sat_azimuths,
            sat_height,
            sun_altitude,
            sun_azimuth
        )
    )

    earth_radius = lumos.constants.EARTH_RADIUS
    sat_z = earth_radius + sat_height

    # Satellite-to-observer distance
    distance = np.sqrt(
        obs_x**2
        + obs_y**2
        + (obs_z - sat_z)**2
    )

    # Unit vector from satellite to observer
    sat_to_obs_x = obs_x / distance
    sat_to_obs_y = obs_y / distance
    sat_to_obs_z = (obs_z - sat_z) / distance

    # Unit vector from satellite to Sun
    sat_to_sun_x = 0
    sat_to_sun_y = np.cos(angle_past_terminator)
    sat_to_sun_z = -np.sin(angle_past_terminator)

    # Solar phase angle:
    # angle between satellite-to-Sun and satellite-to-observer
    cos_phase = (
        sat_to_obs_x * sat_to_sun_x
        + sat_to_obs_y * sat_to_sun_y
        + sat_to_obs_z * sat_to_sun_z
    )

    phase_angle = np.arccos(
        np.clip(cos_phase, -1, 1)
    )

    # Lambertian-sphere phase function
    phase_function = (
        (np.pi - phase_angle) * np.cos(phase_angle)
        + np.sin(phase_angle)
    )

    # Equation 30 from Fankhauser et al.
    intensity = (
        lumos.constants.SUN_INTENSITY
        / distance**2
        * (2 * rho_area_product / (3 * np.pi**2))
        * phase_function
    )

    # Remove satellites inside Earth's shadow
    horizon_angle = np.arccos(
        earth_radius / (earth_radius + sat_height)
    )

    inside_earth_shadow = (
        angle_past_terminator > horizon_angle
    )

    intensity = np.where(
        inside_earth_shadow,
        0,
        intensity
    )

    return intensity


# ------------------------------------------------------------
# Generate Figure 8
# ------------------------------------------------------------

with plt.style.context("dark_background"):

    fig = plt.figure(figsize=(14, 4.3))

    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.10, 1, 1, 1, 1, 1],
        wspace=0.12
    )

    colour_axis = fig.add_subplot(grid[0])
    axes = [
        fig.add_subplot(grid[i], projection="polar")
        for i in range(1, 6)
    ]

    magnitude_levels = np.arange(4, 8.25, 0.25)

    for axis, sun_altitude in zip(
        axes,
        sun_altitudes
    ):

        intensity = diffuse_sphere_intensity(
            sat_altitudes,
            sat_azimuths,
            sun_altitude
        )

        ab_magnitude = (
            lumos.conversions.intensity_to_ab_mag(
                intensity
            )
        )

        contour = axis.contourf(
            np.deg2rad(sat_azimuths),
            90 - sat_altitudes,
            ab_magnitude,
            levels=magnitude_levels,
            cmap="plasma_r",
            norm=matplotlib.colors.Normalize(4, 8),
            extend="both"
        )

        # Centre = zenith, outside edge = horizon
        axis.set_rmax(90)
        axis.set_rticks(
            [10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
        axis.set_yticklabels([])

        # Match paper convention:
        # north at top and east on the left
        axis.set_theta_zero_location("N")

        axis.set_xticks(
            np.deg2rad([0, 90, 180, 270])
        )
        axis.set_xticklabels(["", "", "", ""])

        axis.grid(
            linewidth=0.5,
            alpha=0.30
        )

        axis.set_title(
            f"Sun Alt. = {sun_altitude}°",
            y=-0.18,
            fontsize=11
        )

    colour_bar = plt.colorbar(
        contour,
        cax=colour_axis,
        ticks=[4, 5, 6, 7, 8]
    )

    colour_bar.set_label(
        "AB Magnitude",
        fontsize=12
    )

    colour_bar.ax.invert_yaxis()

    fig.suptitle(
        "Diffuse Sphere Model",
        fontsize=16,
        y=0.97
    )

    plt.show()