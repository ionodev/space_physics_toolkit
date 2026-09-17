import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

f1 = 'MSIS.dat' # Number densities of major neutral species
f2 = 'phot_abs.dat'# Photo absorbtion cross sections
f3 = 'fism_daily_hr19990216.dat' # Incident irradiance

# Column names
f1_columns = ["Height", "O", "N2", "O2", "Mass_density", "Temperature_neutral"]
f2_columns = ["Wavelength", "N2", "O", "O2"]
f3_columns = ["Time", "Wavelength", "Irradiance", "Uncertainty"]

# Read the data into a Pandas DataFrame
df_1 = pd.read_csv(f1, skiprows=18, delim_whitespace=True, names=f1_columns)
df_2 = pd.read_csv(f2, skiprows=8, delim_whitespace=True, names=f2_columns)
df_3 = pd.read_csv(f3, skiprows=1, sep=",", names=f3_columns)

# units:
A = 1e-10  # m
c = 2.998e8  # m/s
h = 6.626e-34  # J s

# Define solar zenith angles
solar_zenith_angles = [0, 15, 30, 45, 60, 75, 85]

# Access data using column names
altitude_orig = df_1["Height"] * 1e3  # m
o_density_orig = df_1["O"] * 1e6  # m-3
o2_density_orig = df_1["O2"] * 1e6  # m-3
n2_density_orig = df_1["N2"] * 1e6  # m-3

wavelength_orig = df_2["Wavelength"]  # m
abs_c_o_orig = df_2["O"]  # m^2
abs_c_o2_orig = df_2["O2"]  # m^2
abs_c_n2_orig = df_2["N2"]  # m^2

# Wavelength and altitude arrays retrieved from abs. cross. data (f2):
wavelength = np.linspace(wavelength_orig.min(), wavelength_orig.max(), 420)
altitude = np.linspace(80e3, 500e3, 420)

# Absorbtion cross section arrays for N2, O, O2, quadratic interpolation
abs_c_n2 = np.clip(
    interp1d(wavelength_orig, abs_c_n2_orig, kind='quadratic')(wavelength), 0, None
)
abs_c_o = np.clip(
    interp1d(wavelength_orig, abs_c_o_orig, kind='quadratic')(wavelength), 0, None
)
abs_c_o2 = np.clip(
    interp1d(wavelength_orig, abs_c_o2_orig, kind='quadratic')(wavelength), 0, None
)

# List for accessing the absorbtion cross sections
abs_cross_sections = [abs_c_n2, abs_c_o2, abs_c_o]

# Wavelength an irradiance data from fism_daily
wavelength_irr = df_3["Wavelength"] * 1e-9  # m
irradiance_inf_orig = df_3["Irradiance"] * wavelength_irr * 0.1 / (
    h * c
)  # photons/m^2/Å/s

# The irradiance data is interpolated to match the shape of the fism_daily wavelength data.
irradiance_inf = interp1d(wavelength_irr, irradiance_inf_orig, kind='linear')(
    wavelength
)

# Interpolate densities up to the maximum altitude in the original data
n2_density = interp1d(altitude_orig, n2_density_orig, kind='linear')(altitude)
o2_density = interp1d(altitude_orig, o2_density_orig, kind='linear')(altitude)
o_density = interp1d(altitude_orig, o_density_orig, kind='linear')(altitude)
densities = [n2_density, o2_density, o_density]


# --- Create PDF to store plots ---
pdf_pages = PdfPages("optical_depth_and_flux_plots.pdf")

# --- First plot: Density vs Altitude ---
fig_density, ax_density = plt.subplots()
ax_density.plot(n2_density, altitude / 1e3, label='N2')
ax_density.plot(o2_density, altitude / 1e3, label='O2')
ax_density.plot(o_density, altitude / 1e3, label='O')
ax_density.legend()
ax_density.set_xscale('log')
ax_density.set_xlabel(r'Density [m$^{-3}$]')
ax_density.set_ylabel(r'Altitude [km]')
fig_density.tight_layout()
pdf_pages.savefig(fig_density)
plt.close(fig_density)

# --- Second plot: Wavelength vs Cross-section ---
fig_cross_section, ax_cross_section = plt.subplots()
ax_cross_section.plot(wavelength / A, abs_c_n2 * 1e4, label='N2')
ax_cross_section.plot(wavelength / A, abs_c_o2 * 1e4, label='O2')
ax_cross_section.plot(wavelength / A, abs_c_o * 1e4, label='O')
ax_cross_section.legend()
ax_cross_section.set_yscale('log')
ax_cross_section.set_xlabel(r'Wavelength [Å]')
ax_cross_section.set_ylabel(r'Cross-section [cm$^2$]')
fig_cross_section.tight_layout()
pdf_pages.savefig(fig_cross_section)
plt.close(fig_cross_section)


def calculate_optical_depth(wavelength, altitude, densities, abs_cross_sections, X):
    """
    Calculates the optical depth, accounting for Earth's curvature.

    Args:
        wavelength (np.ndarray): Array of wavelength values (m).
        altitude (np.ndarray): Array of altitude values for calculation (m).
        densities (list): List of density profiles for each species (m^-3).
        abs_cross_sections (list): List of absorption cross-section profiles for each species (m^2).
        X (float): Solar zenith angle (degrees).

    Returns:
        np.ndarray: 2D array of optical depth values [altitude, wavelength].
    """

    R_earth = 6371e3  # [m] Earth's radius in meters
    tau = np.zeros((len(altitude), len(wavelength))) # optical depth array
    max_altitude = altitude.max() # highest altitude

    # Calculates the optical depth integral, eq. 2.2.4, Rees
    for i, z0 in enumerate(altitude):
        for j, wl in enumerate(wavelength):
            integral_sum = 0
            for k in range(len(densities)):
                mask = (altitude >= z0) & (altitude <= max_altitude)
                altitude_masked = altitude[mask]
                density_masked = densities[k][mask]

                if len(altitude_masked) > 1:
                    sec_chi = (
                        1
                        - ((R_earth + z0) / (R_earth + altitude_masked)) ** 2
                        * np.sin(np.radians(X)) ** 2
                    ) ** (-0.5)
                    integrand = density_masked * sec_chi
                    integral = simpson(integrand, x=altitude_masked)
                    integral_sum += abs_cross_sections[k][j] * integral

            tau[i, j] = integral_sum

    return tau


# For exporting the solar EUV flux data (to be used in prog. task 3)
def export_data(altitude_data, wavelength_data, irradiance_data, X, filename="irradiance_X{}.txt"):
    with open(filename.format(X), 'w') as f:
        # Write header row
        f.write("Altitude [m]\tWavelength [m]\tIrradiance [photons/m^2/A/s]\n")

        for i in range(irradiance_data.shape[0]):
            altitude_m = altitude_data[i]
            wavelength_m = wavelength_data
            irradiance_row = irradiance_data[i]

            for j in range(irradiance_row.shape[0]):
                irradiance_value = irradiance_row[j]
                line = f"{altitude_m}\t{wavelength_m[j]}\t{irradiance_value}\n"
                f.write(line)

    print(f"Data for X={X} exported to '{filename.format(X)}' successfully.")


# --- Store plots for ordered saving ---
optical_depth_figs = []
euv_flux_figs = []

# --- Loop through solar zenith angles ---
for X in solar_zenith_angles:
    # --- Calculate Optical Depth ---
    optical_depth = calculate_optical_depth(
        wavelength, altitude, densities, abs_cross_sections, X
    )
    tau_min = 0.01

    # --- Plotting Optical Depth ---
    fig_tau, ax_tau = plt.subplots()

    img_tau = ax_tau.pcolormesh(
        wavelength / A,
        altitude / 1e3,
        np.clip(optical_depth, tau_min, None),
        shading='auto',
        cmap='inferno_r',
        norm=LogNorm(vmin=tau_min, vmax=5e4),
        rasterized=True
    )

    cbar_tau = fig_tau.colorbar(img_tau, ax=ax_tau, label=r'$\tau$')

    ax_tau.set_xlabel('Wavelength (Å)')
    ax_tau.set_ylabel('Altitude (km)')
    ax_tau.set_title(r'Optical Depth, $\chi_0$ = ' + str(X))
    fig_tau.tight_layout()
    optical_depth_figs.append(fig_tau)


    # EUV flux
    irradiance = irradiance_inf * np.exp(-np.clip(optical_depth, tau_min, None))
    irradiance_clipped = np.clip(irradiance, tau_min, None)

    # --- Plotting EUV Flux ---
    fig_euv, ax_euv = plt.subplots()

    img_euv = ax_euv.pcolormesh(
        wavelength / A,
        altitude / 1e3,
        irradiance_clipped,
        shading='auto',
        cmap='inferno',
        norm=LogNorm(vmin=tau_min),
        rasterized=True
    )

    cbar_euv = fig_euv.colorbar(img_euv, ax=ax_euv, label=r'photons/$m^2$/A/s')

    ax_euv.set_xlabel('Wavelength (Å)')
    ax_euv.set_ylabel('Altitude (km)')
    ax_euv.set_title(r'Solar EUV-flux, $\chi_0$ = ' + str(X))
    fig_euv.tight_layout()
    euv_flux_figs.append(fig_euv)

    # --- Export Data for the current X ---
    #export_data(altitude, wavelength, irradiance_clipped, X)

# --- Save all optical depth plots ---
for fig in optical_depth_figs:
    pdf_pages.savefig(fig)
    plt.close(fig)

# --- Save all Solar EUV flux plots ---
for fig in euv_flux_figs:
    pdf_pages.savefig(fig)
    plt.close(fig)

pdf_pages.close()  # Close the PDF file