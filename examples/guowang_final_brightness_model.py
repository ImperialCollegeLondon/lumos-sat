"""
Final Guowang / HULIANWANG DIGUI Optical Brightness Model

Surface-based brightness model for HULIANWANG DIGUI-01 to DIGUI-10,
implemented with Lumos-Sat.

Final configuration:
- Vertex-defined trapezoidal-prism body
- Binomial BRDF for the spacecraft body
- Two ideal Sun-tracking solar arrays
- Lambertian BRDF (rho = 0.01) for the solar arrays
- Direct sunlight and Earthshine included
- Earthshine represented using a Phong Earth BRDF
- Reflected intensity converted to AB magnitude

The same final model is applied to all ten satellites without
satellite-specific retuning.

The spacecraft geometry, array attitude law, and BRDF parameters are
effective modelling assumptions rather than confirmed Guowang design
or measured material properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from lumos.geometry import Surface
from lumos.brdf.library import LAMBERTIAN, BINOMIAL, PHONG

import lumos.calculator
import lumos.conversions


#------------------------------------------------------------
# 1. VERTEX-BASED TRAPEZOID GEOMETRY VALUES
#------------------------------------------------------------

# Coordinate system assumption:
# x = left/right direction
# y = front/back direction
# z = vertical direction
#
# z = 0 is the Earth-facing bottom of the satellite body
# +z points away from Earth
# -z points towards Earth

# Geometry 
bottom_width = 1.60     # m, wider Earth-facing bottom width
top_width = 1.00        # m, smaller top width
body_length = 2.60      # m, front-back length/depth
body_height = 1.40      # m, body height

# Solar array area:
# each solar array = 3 m x 2 m = 6 m^2
A_array = 6.0           # m^2, one solar array


#------------------------------------------------------------
# EARTHSHINE SETTINGS
#------------------------------------------------------------

# Lumos-Sat requires a callable BRDF for Earth's surface when
# include_earthshine=True.
#
# Representative generic-vegetation Earth BRDF from:
# Fankhauser, Tyson & Askari (2023), Satellite Optical Brightness
#
# Phong parameters:
#   Kd = 0.53
#   Ks = 0.28
#   n  = 7.31
#
# The paper used a 151 x 151 Earthshine patch discretisation.
#
EARTH_BRDF = PHONG(
    Kd=0.53,
    Ks=0.28,
    n=7.31
)

EARTH_PANEL_DENSITY = 151

# Range-normalisation settings for the pooled phase-angle plot
EARTH_RADIUS_KM = 6371.0
REFERENCE_RANGE_KM = 1000.0


#------------------------------------------------------------
# 2. HELPER FUNCTIONS
#------------------------------------------------------------

def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Cannot normalise zero vector.")

    return vector / norm


def polygon_area_and_normal(vertices, expected_normal=None):
    """
    Calculate area and normal of a planar polygon from its vertices.

    The face is split into triangles using the first vertex.
    The cross products give both area and normal direction.

    expected_normal is used to flip the normal if it points the wrong way.
    """

    vertices = np.asarray(vertices, dtype=float)

    if len(vertices) < 3:
        raise ValueError("A face needs at least 3 vertices.")

    normal_sum = np.zeros(3)
    v0 = vertices[0]

    for i in range(1, len(vertices) - 1):
        edge1 = vertices[i] - v0
        edge2 = vertices[i + 1] - v0

        normal_sum += np.cross(edge1, edge2)

    area = 0.5 * np.linalg.norm(normal_sum)
    normal = unit_vector(normal_sum)

    if expected_normal is not None:
        expected_normal = unit_vector(expected_normal)

        if np.dot(normal, expected_normal) < 0:
            normal = -normal

    return area, normal


def sun_vector_lumos(angle_past_terminator):
    alpha = angle_past_terminator

    return unit_vector(
        np.array([
            0.0,
            np.cos(alpha),
            -np.sin(alpha)
        ])
    )


def panel_normal_from_psi(psi):
    return unit_vector(
        np.array([
            0.0,
            np.sin(psi),
            np.cos(psi)
        ])
    )


def analytic_sun_tracking_normal(angle_past_terminator):
    sun_vector = sun_vector_lumos(angle_past_terminator)

    sx, sy, sz = sun_vector

    psi = np.arctan2(sy, sz)

    return panel_normal_from_psi(psi)


def analytic_backside_normal(angle_past_terminator):
    return -analytic_sun_tracking_normal(angle_past_terminator)


def calculate_slant_range_km(
    sat_altitude_km,
    elevation_deg,
    earth_radius_km=EARTH_RADIUS_KM
):
    """
    Calculate observer-to-satellite slant range from satellite altitude
    above the Earth and observer-frame elevation angle.

    This is the range needed for brightness normalisation. It is NOT the
    same as the satellite altitude above the Earth's surface.

    """

    sat_altitude_km = np.asarray(sat_altitude_km, dtype=float)
    elevation_rad = np.deg2rad(
        np.asarray(elevation_deg, dtype=float)
    )

    satellite_radius_km = earth_radius_km + sat_altitude_km

    square_root_term = (
        satellite_radius_km ** 2
        - (earth_radius_km * np.cos(elevation_rad)) ** 2
    )

    # Guard against tiny negative values caused by floating-point rounding.
    square_root_term = np.maximum(square_root_term, 0.0)

    slant_range_km = (
        -earth_radius_km * np.sin(elevation_rad)
        + np.sqrt(square_root_term)
    )

    return slant_range_km


def normalise_magnitude_to_range(
    magnitude,
    slant_range_km,
    reference_range_km=REFERENCE_RANGE_KM
):
    """
    Convert apparent magnitude at the actual observer range to the
    magnitude the same observation would have at a standard range.

    m_ref = m_app - 5*log10(R / R_ref)

    The phase angle and spacecraft orientation are NOT changed.
    Only the inverse-square observer-distance effect is removed.

    """

    magnitude = np.asarray(magnitude, dtype=float)
    slant_range_km = np.asarray(slant_range_km, dtype=float)

    normalised = np.full_like(
        magnitude,
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(magnitude)
        & np.isfinite(slant_range_km)
        & (slant_range_km > 0.0)
    )

    normalised[valid] = (
        magnitude[valid]
        - 5.0 * np.log10(
            slant_range_km[valid] / reference_range_km
        )
    )

    return normalised


#------------------------------------------------------------
# 3. DEFINE BODY VERTICES
#------------------------------------------------------------

def define_trapezoid_vertices():
    """
    Defines a simple trapezoidal prism body.

    Bottom face is wider and Earth-facing.
    Top face is narrower and points away from Earth.
    Side faces naturally slope inwards because the top width is smaller.

    """

    Wb = bottom_width
    Wt = top_width
    L = body_length
    H = body_height

    # Bottom vertices, z = 0
    b_front_left = np.array([-Wb / 2,  L / 2, 0.0])
    b_front_right = np.array([ Wb / 2,  L / 2, 0.0])
    b_back_right = np.array([ Wb / 2, -L / 2, 0.0])
    b_back_left = np.array([-Wb / 2, -L / 2, 0.0])

    # Top vertices, z = H, narrower width
    t_front_left = np.array([-Wt / 2,  L / 2, H])
    t_front_right = np.array([ Wt / 2,  L / 2, H])
    t_back_right = np.array([ Wt / 2, -L / 2, H])
    t_back_left = np.array([-Wt / 2, -L / 2, H])

    return {
        "b_front_left": b_front_left,
        "b_front_right": b_front_right,
        "b_back_right": b_back_right,
        "b_back_left": b_back_left,
        "t_front_left": t_front_left,
        "t_front_right": t_front_right,
        "t_back_right": t_back_right,
        "t_back_left": t_back_left
    }


#------------------------------------------------------------
# 4. BUILD BODY FACES FROM VERTICES
#------------------------------------------------------------

def build_body_faces_from_vertices():
    """
    Creates the body faces using vertices.

    Each face has:
    - name
    - vertices
    - expected outward normal direction
    """

    v = define_trapezoid_vertices()

    faces = [
        {
            "name": "bottom",
            "vertices": [
                v["b_front_left"],
                v["b_back_left"],
                v["b_back_right"],
                v["b_front_right"]
            ],
            "expected_normal": np.array([0.0, 0.0, -1.0])
        },
        {
            "name": "top",
            "vertices": [
                v["t_front_left"],
                v["t_front_right"],
                v["t_back_right"],
                v["t_back_left"]
            ],
            "expected_normal": np.array([0.0, 0.0, 1.0])
        },
        {
            "name": "front",
            "vertices": [
                v["b_front_left"],
                v["b_front_right"],
                v["t_front_right"],
                v["t_front_left"]
            ],
            "expected_normal": np.array([0.0, 1.0, 0.0])
        },
        {
            "name": "back",
            "vertices": [
                v["b_back_left"],
                v["t_back_left"],
                v["t_back_right"],
                v["b_back_right"]
            ],
            "expected_normal": np.array([0.0, -1.0, 0.0])
        },
        {
            "name": "left_side",
            "vertices": [
                v["b_front_left"],
                v["t_front_left"],
                v["t_back_left"],
                v["b_back_left"]
            ],
            "expected_normal": np.array([-1.0, 0.0, 0.0])
        },
        {
            "name": "right_side",
            "vertices": [
                v["b_front_right"],
                v["b_back_right"],
                v["t_back_right"],
                v["t_front_right"]
            ],
            "expected_normal": np.array([1.0, 0.0, 0.0])
        }
    ]

    body_faces = []

    print()
    print("TRAPEZOID BODY FACE GEOMETRY")
    print("----------------------------")

    for face in faces:
        area, normal = polygon_area_and_normal(
            face["vertices"],
            expected_normal=face["expected_normal"]
        )

        body_faces.append({
            "name": face["name"],
            "area": area,
            "normal": normal
        })

        print(f"{face['name']}:")
        print(f"  area   = {area:.4f} m^2")
        print(f"  normal = {normal}")

    return body_faces


#------------------------------------------------------------
# 5. BUILD LUMOS-SAT SURFACE MODEL
#------------------------------------------------------------

def build_trapezoid_body_binomial_model():
    """
    Vertex-based trapezoid body + Binomial BRDF model.

    Final model:
    - Body faces use BINOMIAL BRDF
    - Solar arrays use LAMBERTIAN(0.01)

    The solar arrays keep the same dark Lambertian BRDF because the
    dark binomial solar-array test did not give a meaningful improvement.
    
    """

    # ------------------------------------------------------------
    # BODY BINOMIAL PARAMETERS
    # ------------------------------------------------------------
    # These are fitted/test values, not confirmed material properties.
    # ------------------------------------------------------------

    B_body = np.array([
        [-0.47, 0.00]
    ])

    C_body = np.array([
        [-0.50, 0.00]
    ])

    body_brdf = BINOMIAL(
        B=B_body,
        C=C_body,
        d=10,
        l1=0
    )

    # Final solar-array model
    array_brdf = LAMBERTIAN(0.01)

    body_faces = build_body_faces_from_vertices()

    surfaces = []

    # Add body faces
    for face in body_faces:
        surfaces.append(
            Surface(
                area=face["area"],
                normal=face["normal"],
                brdf=body_brdf
            )
        )

    # Add left solar array front/back
    surfaces.append(
        Surface(
            area=A_array,
            normal=analytic_sun_tracking_normal,
            brdf=array_brdf
        )
    )

    surfaces.append(
        Surface(
            area=A_array,
            normal=analytic_backside_normal,
            brdf=array_brdf
        )
    )

    # Add right solar array front/back
    surfaces.append(
        Surface(
            area=A_array,
            normal=analytic_sun_tracking_normal,
            brdf=array_brdf
        )
    )

    surfaces.append(
        Surface(
            area=A_array,
            normal=analytic_backside_normal,
            brdf=array_brdf
        )
    )

    return surfaces


#------------------------------------------------------------
# 6. CALCULATE MODEL MAGNITUDES
#------------------------------------------------------------

def calculate_model_magnitudes(surfaces, dataframe):

    model_mags = []

    for _, row in dataframe.iterrows():

        sat_height_m = row["sat_altitude_km_satchecker"] * 1000.0

        sat_alt = row["alt_deg_satchecker"]
        sat_az = row["az_deg_satchecker"]

        sun_alt = row["solar_elevation_deg_satchecker"]
        sun_az = row["solar_azimuth_deg_satchecker"]

        try:
            intensity = lumos.calculator.get_intensity_observer_frame(
                sat_surfaces=surfaces,
                sat_heights=sat_height_m,
                sat_altitudes=sat_alt,
                sat_azimuths=sat_az,
                sun_altitude=sun_alt,
                sun_azimuth=sun_az,
                include_sun=True,
                include_earthshine=True,
                earth_panel_density=EARTH_PANEL_DENSITY,
                earth_brdf=EARTH_BRDF
            )

            intensity = float(np.asarray(intensity).squeeze())

            if intensity > 0:
                mag = lumos.conversions.intensity_to_ab_mag(intensity)
                mag = float(np.asarray(mag).squeeze())
            else:
                mag = np.nan

        except Exception as error:
            print("Calculation failed:", error)
            mag = np.nan

        model_mags.append(mag)

    return np.array(model_mags)


#------------------------------------------------------------
# 7. INPUT / OUTPUT CONFIGURATION
#------------------------------------------------------------

# Recommended repository layout:
# project/
# ├── guowang_final_brightness_model.py
# ├── data/
# │   ├── HULIANWANG DIGUI-01_observations.csv
# │   ├── HULIANWANG DIGUI-02_observations.csv
# │   └── ...
# └── results/
#
# Change DATA_DIR only if the observation CSV files are stored elsewhere.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

satellite_names = [
    f"HULIANWANG DIGUI-{i:02d}"
    for i in range(1, 11)
]


def find_observation_csv(data_dir, satellite_name):
    """Find the SCORE observation CSV for one DIGUI satellite."""
    satellite_number = int(satellite_name.split("-")[-1])

    matches = []

    for path in data_dir.rglob("*.csv"):
        filename = path.name.lower()

        if (
            "observation" in filename
            and "prediction" not in filename
            and (
                f"digui-{satellite_number:02d}" in filename
                or f"digui_{satellite_number:02d}" in filename
                or f"digui {satellite_number:02d}" in filename
            )
        ):
            matches.append(path)

    if not matches:
        return None

    matches.sort(key=lambda path: (len(path.parts), len(path.name)))
    return matches[0]


required_cols = [
    "apparent_magnitude",
    "apparent_magnitude_uncertainty",
    "phase_angle_deg_satchecker",
    "alt_deg_satchecker",
    "az_deg_satchecker",
    "sat_altitude_km_satchecker",
    "solar_elevation_deg_satchecker",
    "solar_azimuth_deg_satchecker"
]


model_name = "Vertex-based trapezoid body + Binomial body BRDF + Lambertian arrays + Earthshine"
model_column = "vertex_trapezoid_binomial_body_mag"

all_results = []
all_plot_data = []

print()
print("Running final Guowang brightness model on all DIGUI satellites...")
print("Earth BRDF: PHONG(Kd=0.53, Ks=0.28, n=7.31) - generic vegetation")
print(f"Earthshine discretisation: {EARTH_PANEL_DENSITY} x {EARTH_PANEL_DENSITY} patches")
print("===========================================================")

# Build the Lumos-Sat surface model once
surfaces = build_trapezoid_body_binomial_model()


#------------------------------------------------------------
# 8. LOOP THROUGH ALL SATELLITES
#------------------------------------------------------------

for satellite_name in satellite_names:

    csv_path = find_observation_csv(
        DATA_DIR,
        satellite_name
    )

    print()
    print("--------------------------------------------------")
    print(f"Satellite: {satellite_name}")
    print("--------------------------------------------------")

    if csv_path is None:
        print(
            f"Could not find an observation CSV for {satellite_name} "
            f"inside: {DATA_DIR.resolve()}"
        )
        continue

    print("Using CSV:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    original_obs = len(df)

    df = df.dropna(
        subset=required_cols
    ).copy()

    usable_obs = len(df)

    print(f"Original observations: {original_obs}")
    print(f"Usable observations after removing missing geometry: {usable_obs}")

    if usable_obs == 0:
        print("No usable observations, skipping.")
        continue

    # Calculate model magnitudes
    df[model_column] = calculate_model_magnitudes(
        surfaces,
        df
    )

    # --------------------------------------------------------
    # CALCULATE OBSERVER-TO-SATELLITE SLANT RANGE
    # AND NORMALISE BOTH SCORE AND MODEL MAGNITUDES TO 1000 km
    # --------------------------------------------------------
    #
    # The satellite altitude above Earth is NOT the same as the
    # observer-to-satellite range. The slant range is calculated
    # from altitude and observer-frame elevation angle.
    #
    # Both SCORE and model magnitudes receive exactly the same
    # distance correction. Phase angle and spacecraft orientation
    # are left unchanged.
    # --------------------------------------------------------

    df["slant_range_km"] = calculate_slant_range_km(
        df["sat_altitude_km_satchecker"].to_numpy(),
        df["alt_deg_satchecker"].to_numpy()
    )

    df["score_mag_1000km"] = normalise_magnitude_to_range(
        df["apparent_magnitude"].to_numpy(),
        df["slant_range_km"].to_numpy()
    )

    df["model_mag_1000km"] = normalise_magnitude_to_range(
        df[model_column].to_numpy(),
        df["slant_range_km"].to_numpy()
    )

    valid_df = df.dropna(
        subset=[
            "apparent_magnitude",
            model_column
        ]
    ).copy()

    valid_obs = len(valid_df)

    if valid_obs == 0:
        print("No valid model predictions, skipping.")
        continue

    errors = (
        valid_df[model_column]
        - valid_df["apparent_magnitude"]
    )

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    print()
    print("RESULTS")
    print(f"Valid observations: {valid_obs}")
    print(f"MSE:  {mse:.6f} mag^2")
    print(f"RMSE: {rmse:.6f} mag")
    print(f"MAE:  {mae:.6f} mag")
    print(f"Bias/model - SCORE: {bias:.6f} mag")

    # Save individual prediction CSV
    short_name = satellite_name.replace("HULIANWANG ", "").replace("-", "_")

    output_predictions = OUTPUT_DIR / (
        f"{short_name}_predictions.csv"
    )

    df.to_csv(
        output_predictions,
        index=False
    )

    # Save individual plot
    df_plot = df.sort_values(
        "phase_angle_deg_satchecker"
    ).copy()

    # Store this satellite's plotting data so that all DIGUI
    # results can also be shown together in one combined figure.
    all_plot_data.append(
        {
            "satellite_name": satellite_name,
            "short_name": short_name,
            "df_plot": df_plot.copy()
        }
    )

    plt.figure(figsize=(11, 7))

    plt.scatter(
        df_plot["phase_angle_deg_satchecker"],
        df_plot[model_column],
        color="orange",
        s=45,
        label="Model + Earthshine",
        zorder=3
    )

    plt.errorbar(
        df_plot["phase_angle_deg_satchecker"],
        df_plot["apparent_magnitude"],
        yerr=df_plot["apparent_magnitude_uncertainty"],
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.1,
        capsize=3,
        alpha=0.85,
        label="SCORE observations",
        zorder=2
    )

    plt.xlabel("Phase angle (degrees)")
    plt.ylabel("Apparent magnitude")

    plt.title(
        f"{satellite_name}: SCORE vs binomial body model with Earthshine"
    )

    # Magnitude scale: lower magnitude means brighter
    plt.gca().invert_yaxis()

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    output_plot = OUTPUT_DIR / (
        f"{short_name}_comparison.png"
    )

    plt.savefig(
        output_plot,
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(output_predictions)
    print(output_plot)

    # Add to combined results table
    all_results.append(
        {
            "satellite": satellite_name,
            "model": model_name,
            "original_observations": original_obs,
            "usable_observations_after_dropna": usable_obs,
            "valid_model_observations": valid_obs,
            "mse_mag2": mse,
            "rmse_mag": rmse,
            "mae_mag": mae,
            "bias_model_minus_score_mag": bias,
            "body_B": "[[-0.47, 0.00]]",
            "body_C": "[[-0.50, 0.00]]",
            "body_d": 10,
            "body_l1": 0,
            "array_brdf": "LAMBERTIAN(0.01)",
            "include_earthshine": True,
            "earth_brdf": "PHONG(Kd=0.53, Ks=0.28, n=7.31) - generic vegetation",
            "earth_panel_density": EARTH_PANEL_DENSITY
        }
    )


#------------------------------------------------------------
# 9. SAVE COMBINED RESULTS
#------------------------------------------------------------

results_df = pd.DataFrame(all_results)

if len(results_df) > 0:

    results_df = results_df.sort_values(
        "mse_mag2"
    ).reset_index(drop=True)

    output_results = OUTPUT_DIR / "all_DIGUI_results.csv"

    results_df.to_csv(
        output_results,
        index=False
    )

    print()
    print("===========================================================")
    print("FINAL COMBINED RESULTS")
    print("===========================================================")
    print(results_df)

    print()
    print("Saved combined results:")
    print(output_results)

else:
    print()
    print("No satellites were successfully processed.")
    print("Check the CSV file paths above.")


#------------------------------------------------------------
# 10. SAVE COMBINED 5 x 2 SUBPLOT FIGURE
#------------------------------------------------------------

if len(all_plot_data) > 0:

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(14, 22)
    )

    axes = axes.flatten()

    for ax, plot_info in zip(axes, all_plot_data):

        df_plot = plot_info["df_plot"]
        short_name = plot_info["short_name"]

        # Model predictions
        ax.scatter(
            df_plot["phase_angle_deg_satchecker"],
            df_plot[model_column],
            color="orange",
            s=25,
            label="Model",
            zorder=3
        )

        # SCORE observations and uncertainties
        ax.errorbar(
            df_plot["phase_angle_deg_satchecker"],
            df_plot["apparent_magnitude"],
            yerr=df_plot["apparent_magnitude_uncertainty"],
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.0,
            capsize=2,
            alpha=0.85,
            label="SCORE",
            zorder=2
        )

        ax.set_title(short_name, fontsize=11)
        ax.set_xlabel("Phase angle (degrees)")
        ax.set_ylabel("Apparent magnitude")

        # Magnitude scale: lower magnitude means brighter
        ax.invert_yaxis()

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide any unused panels if fewer than 10 satellites are processed.
    for ax in axes[len(all_plot_data):]:
        ax.axis("off")

    fig.suptitle(
        "All DIGUI satellites: Earthshine model vs SCORE observations",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    output_combined_subplot = (
        OUTPUT_DIR / "all_DIGUI_comparisons.png"
    )

    plt.savefig(
        output_combined_subplot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Saved combined subplot figure:")
    print(output_combined_subplot)

else:
    print()
    print("No plotting data available for combined subplot figure.")



#------------------------------------------------------------
# 11. SAVE POOLED ALL-DIGUI RANGE-NORMALISED PHASE-ANGLE PLOT
#------------------------------------------------------------

if len(all_plot_data) > 0:

    # Combine every successfully processed DIGUI satellite into
    # one dataframe. Each row remains one real SCORE observation
    # with its corresponding model prediction.
    pooled_df = pd.concat(
        [
            plot_info["df_plot"].assign(
                satellite=plot_info["satellite_name"]
            )
            for plot_info in all_plot_data
        ],
        ignore_index=True
    )

    pooled_score = pooled_df.dropna(
        subset=[
            "phase_angle_deg_satchecker",
            "score_mag_1000km"
        ]
    ).copy()

    pooled_model = pooled_df.dropna(
        subset=[
            "phase_angle_deg_satchecker",
            "model_mag_1000km"
        ]
    ).copy()

    plt.figure(
        figsize=(12, 8)
    )

    # SCORE observations:
    # Keep uncertainty bars, but make them light so the pooled
    # figure does not become too cluttered.
    
    plt.errorbar(
        pooled_score["phase_angle_deg_satchecker"],
        pooled_score["score_mag_1000km"],
        yerr=pooled_score["apparent_magnitude_uncertainty"],
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
        markersize=4,
        elinewidth=0.6,
        capsize=0,
        alpha=0.45,
        label="SCORE observations",
        zorder=2
    )

    # Model predictions
    plt.scatter(
        pooled_model["phase_angle_deg_satchecker"],
        pooled_model["model_mag_1000km"],
        color="orange",
        s=24,
        alpha=0.60,
        label="Model + Earthshine",
        zorder=3
    )

    plt.xlabel(
        "Phase angle (degrees)"
    )

    plt.ylabel(
        "Range-normalised apparent magnitude (1000 km)"
    )

    plt.title(
        "All DIGUI satellites: Earthshine model vs SCORE observations (1000 km normalised)"
    )

    # Astronomical magnitude convention:
    # smaller magnitude = brighter
    plt.gca().invert_yaxis()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        fontsize=9
    )

    plt.tight_layout()

    output_pooled_plot = (
        OUTPUT_DIR / "all_DIGUI_range_normalised_1000km.png"
    )

    plt.savefig(
        output_pooled_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Saved pooled all-DIGUI 1000 km range-normalised plot:")
    print(output_pooled_plot)
    print(
        f"Pooled SCORE observations plotted: {len(pooled_score)}"
    )
    print(
        f"Pooled model predictions plotted: {len(pooled_model)}"
    )

else:
    print()
    print(
        "No plotting data available for pooled "
        "range-normalised phase-angle figure."
    )


# Run from the repository root with:
# python guowang_final_brightness_model.py
