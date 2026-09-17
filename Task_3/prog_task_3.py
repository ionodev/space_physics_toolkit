import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from scipy.integrate import simpson
from matplotlib.widgets import Slider
from matplotlib.backends.backend_pdf import PdfPages

def read_data(filename):
    # Read optical depth and solar EUV data (exported from prog task 2)
    df = pd.read_csv(filename, sep='\t', skiprows=[0], header=None)
    df.columns = ['Altitude [m]', 'Wavelength [m]', 'Irradiance [photons/m^2/A/s]']
    altitude = df['Altitude [m]'].unique()
    wavelength = df['Wavelength [m]'].unique()
    irradiance_2d = np.zeros((len(altitude), len(wavelength)))
    for index, row in df.iterrows():
        alt_index = np.where(altitude == row['Altitude [m]'])[0][0]
        wave_index = np.where(wavelength == row['Wavelength [m]'])[0][0]
        irradiance_2d[alt_index, wave_index] = row['Irradiance [photons/m^2/A/s]']
    print(f"Data read from '{filename}' successfully.")
    return altitude, wavelength, irradiance_2d

# Read MSIS and photoionization data
f1 = 'MSIS.dat'
f2 = 'phot_ion.dat'

f1_columns = ["Height", "O", "N2", "O2", "Mass_density", "Temperature_neutral"]
f2_columns = ["Wavelength", "N2", "O", "O2"]
df_1 = pd.read_csv(f1, skiprows=18, delim_whitespace=True, names=f1_columns)
df_2 = pd.read_csv(f2, skiprows=6, delim_whitespace=True, names=f2_columns)

# Original density, cross-section, altitude and wavelength values
altitude_orig = df_1["Height"] * 1e3  # m
o_density_orig = df_1["O"] * 1e6  # m-3
o2_density_orig = df_1["O2"] * 1e6  # m-3
n2_density_orig = df_1["N2"] * 1e6  # m-3

wavelength_orig = df_2["Wavelength"]  # m
ion_cs_o_orig = df_2["O"]  # m^2
ion_cs_o2_orig = df_2["O2"]  # m^2
ion_cs_n2_orig = df_2["N2"]  # m^2

A = 1e-10  # m Ångstrøm

# Read irradiance data for different solar zenith angles
solar_zenith_angles = [0, 15, 30, 45, 60, 75, 85]
irradiance_data = {}
for angle in solar_zenith_angles:
    filename = f"irradiance_X{angle}.txt"
    try:
        altitude_irradiance, wavelength_irradiance_, irradiance = read_data(filename)
        # Ensure wavelength is consistent and in meters
        wavelength_irradiance = np.clip(wavelength_irradiance_ * A, wavelength_orig.min(), wavelength_orig.max())
        irradiance_data[angle] = irradiance
        if angle == 0:
            altitude = altitude_irradiance
            wavelength = wavelength_irradiance
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found.")

# Interpolated ionization cross-sections
ion_cs_n2 = np.clip(interp1d(wavelength_orig, ion_cs_n2_orig, kind='cubic')(wavelength), 0, None)
ion_cs_o = np.clip(interp1d(wavelength_orig, ion_cs_o_orig, kind='cubic')(wavelength), 0, None)
ion_cs_o2 = np.clip(interp1d(wavelength_orig, ion_cs_o2_orig, kind='cubic')(wavelength), 0, None)
ion_cross_sections = np.array([ion_cs_n2, ion_cs_o2, ion_cs_o])

# Interpolated densities
n2_density = interp1d(altitude_orig, n2_density_orig, kind='linear')(altitude)
o2_density = interp1d(altitude_orig, o2_density_orig, kind='linear')(altitude)
o_density = interp1d(altitude_orig, o_density_orig, kind='linear')(altitude)
densities = np.array([n2_density, o2_density, o_density])

### constants
c = 2.998e8  # m/s
h = 6.626e-34  # J s
q_e = 1.602e-19  # J

E_th_n2 = 15.58  # eV
E_th_o2 = 12.08  # eV
E_th_o = 13.61  # eV

lambda_th_n2 = h * c / (E_th_n2 * q_e) # m
lambda_th_o2 = h * c / (E_th_o2 * q_e) # m
lambda_th_o = h * c / (E_th_o * q_e) # m

E_th_values = [E_th_n2, E_th_o2, E_th_o] # eV

E_values = np.arange(0, 101, 1)  # Energy steps from 0 to 100 eV

def ionization_wavelengths(E_values, E_th_values):
    """Calculates ionization wavelengths for given energy and threshold energies."""
    wavelength_array = np.empty(3, dtype=object)
    for i, E_th in enumerate(E_th_values):
        wavelength_j = h * c / ((E_values + E_th) * q_e)
        wavelength_array[i] = wavelength_j  # m
    return wavelength_array

wavelength_ions = ionization_wavelengths(E_values, E_th_values)  # m
d_lambda = wavelength[1] - wavelength[0]  # m (stepsize is approximately constant)

wavelengths_selected = np.empty(3, dtype=object)
mask_array = np.empty(3, dtype=object)

# Selects wavelengths corresponding to ionization energy
for i in range(len(wavelength_ions)):
    wavelengths_selected_i = []
    for wl_i in wavelength_ions[i]:
        for wl in wavelength:
            if abs(wl_i / A - wl / A) <= d_lambda / A / 2:
                wavelengths_selected_i.append(wl)
    wavelengths_selected[i] = np.array(wavelengths_selected_i)
    mask = np.isin(wavelength, wavelengths_selected[i])
    mask_array[i] = mask

def calculate_production_rate(E_values, altitude, irradiance, cross_sections, densities, mask):
    """Calculates the production rate"""
    P = np.zeros((len(altitude), len(E_values)))
    for i in range(len(altitude)):
        sum_val = 0
        for k in range(len(E_values)):
            for j in range(len(densities)):
                contribution = densities[j][i] * cross_sections[j][mask[j]] * irradiance[i][mask[j]] * d_lambda
                sum_val += np.sum(contribution)
            P[i, k] = sum_val
    return P

def calculate_total_photoionization_rate(altitude, densities, irradiance, cross_sections, wavelength, E_th_values, h, c, q_e):
    """
    Calculates the total photoionization rate
    """
    n_species = densities.shape[0] # m^-3
    n_altitudes = altitude.shape[0] # m
    n_wavelengths = wavelength.shape[0] # m

    lambda_th = h * c / (np.array(E_th_values) * q_e)  # Threshold wavelengths. Shape: (n_species,)

    total_ionization_rate = np.zeros(n_altitudes)

    for j in range(n_species):
        # Create a mask for valid wavelengths FOR THIS SPECIES
        valid_wavelengths = wavelength <= lambda_th[j]  # Shape: (n_wavelengths,)

        # Use np.where to get the *indices* of the valid wavelengths
        valid_indices = np.where(valid_wavelengths)[0]

        if valid_indices.size > 1:
            # Index the cross_sections and irradiance using the valid indices
            integrand = irradiance[:, valid_indices] * cross_sections[j, valid_indices]
            # Shape: (n_altitudes, n_valid_wavelengths_for_this_species)

            # Integrate using Simpson's rule along the wavelength axis
            integral = simpson(integrand, wavelength[valid_indices], axis=1)  # Shape: (n_altitudes,)

            # Multiply by densities and add to the total
            total_ionization_rate += densities[j] * integral
        elif valid_indices.size == 1:
            integrand = irradiance[:, valid_indices] * cross_sections[j, valid_indices]
            total_ionization_rate += densities[j] * integrand.flatten()
        else:
            pass  # No valid wavelengths for this species

    return total_ionization_rate

# Calculate total photo-ionization rate for each solar zenith angle
q_values = {}
for angle in solar_zenith_angles:
    if angle in irradiance_data:
        q_values[angle] = calculate_total_photoionization_rate(
            altitude, densities, irradiance_data[angle] / A, ion_cross_sections, wavelength, E_th_values, h, c, q_e)

# --- Chapman Curve Implementation with Variable Scale Height ---
scale_heights = {
    0: 25000,
    15: 25000,
    30: 27000,
    45: 30000,
    60: 36000,
    75: 36000,
    85: 34000,
}

# Find the altitude of the peak ionization rate for X=0 to use as a reference
if 0 in q_values:
    q_0_max = np.max(q_values[0])
    z_0_max_ref = altitude[np.argmax(q_values[0])]
else:
    print("Warning: No data for solar zenith angle 0 to determine z_0_max.")
    z_0_max_ref = np.mean(altitude) # Use a default if no data for X=0

def q_m_chapman(X, q_0_max):
    """Calculates the peak ionization rate based on Chapman theory."""
    return q_0_max * np.cos(np.radians(X))

def z_m_chapman(X, z_0_max, H):
    """Calculates the altitude of the peak ionization rate based on Chapman theory."""
    return z_0_max + H * np.log(1 / np.cos(np.radians(X)))

def chapman_profile(z, X, q_0_max, z_0_max_ref, scale_heights):
    """Calculates the Chapman profile for a given solar zenith angle and altitude.

    Args:
        z (np.ndarray): Array of altitudes.
        X (int): Solar zenith angle.
        q_0_max (float): Peak ionization rate at X=0.
        z_0_max_ref (float): Altitude of peak ionization rate at X=0.
        scale_heights (dict): Dictionary of scale heights for each angle.

    Returns:
        np.ndarray: Array of ionization rates at the given altitudes.
    """
    if X not in scale_heights:
        print(f"Warning: Scale height not defined for angle {X}. Using default.")
        H = scale_heights.get(0, 25000) # Default to the scale height at 0 degrees or 25000
    else:
        H = scale_heights[X]

    zm = z_m_chapman(X, z_0_max_ref, H)
    qm = q_m_chapman(X, q_0_max)
    chi = (z - zm) / H
    return qm * np.exp(1 - chi - np.exp(-chi))

# --- Generating PDF ---
pdf_pages = PdfPages('old/production_and_ionization_profiles.pdf')

# --- Production Rate Plots for each solar zenith angle ---
for angle in solar_zenith_angles:
    if angle in irradiance_data:
        P = calculate_production_rate(E_values, altitude, irradiance_data[angle] / A, ion_cross_sections, densities, mask_array)

        fig, ax = plt.subplots()
        img = ax.pcolormesh(E_values, altitude / 1e3, P, shading='auto', cmap='inferno', norm=LogNorm(1e-2, 1e11), rasterized=True)
        cbar = fig.colorbar(img, ax=ax, label=r'photo-electrons/m$^3$/s')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title(r'Photo-Electron Production Rate, $\chi_0$ = ' + str(angle))
        pdf_pages.savefig(fig)
        plt.close(fig)

# --- Total Photoionization Rate Plot ---
fig_q, ax_q = plt.subplots()
for angle, q in q_values.items():
    ax_q.plot(q, altitude / 1e3, label=r'$\chi_0 =$' + f'{angle}')
ax_q.set_xlabel(r"Ionization Rate (ions/m$^{3}$/s)")
ax_q.set_ylabel("Altitude (km)")
ax_q.set_xscale('log')
ax_q.set_xlim(1e-1, 1e10)
ax_q.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=13))
ax_q.set_title("Total Photo-ionization Rate vs. Altitude")
ax_q.grid(True, linestyle=':')
ax_q.legend()
pdf_pages.savefig(fig_q)
plt.close(fig_q)

# --- Chapman Curve Plots ---
fig_chapman, ax_chapman = plt.subplots()
altitude_km = altitude / 1e3
for X in solar_zenith_angles:
    if 0 in q_values:
        chapman_q = chapman_profile(altitude, X, q_0_max, z_0_max_ref, scale_heights)
        ax_chapman.plot(chapman_q, altitude_km, label=r'${\chi_0} =$' + f'{X} (H={scale_heights.get(X, scale_heights.get(0, 25000))/1e3:.0f} km)')
    else:
        print("Cannot plot Chapman curves as q_0 is not available.")
        break
ax_chapman.set_xlabel(r"Ionization Rate (ions/m$^{3}$/s)")
ax_chapman.set_ylabel("Altitude (km)")
ax_chapman.set_xscale('log')
ax_chapman.set_title("Total Photo-ionization Rate vs. Altitude (Chapman)")
ax_chapman.set_xlim(1e-1, 1e10)
ax_chapman.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=13))
ax_chapman.grid(True, linestyle=':')
ax_chapman.legend()
pdf_pages.savefig(fig_chapman)
plt.close(fig_chapman)

# --- Comparing Peak Ionization Rates ---
if 0 in q_values:
    q_values_max = {angle: np.max(q) for angle, q in q_values.items()}
    chapman_q_peak = {angle: q_m_chapman(angle, q_0_max) for angle in solar_zenith_angles}

    fig_peak_q, ax_peak_q = plt.subplots()
    ax_peak_q.grid(True, linestyle=':')
    ax_peak_q.scatter(solar_zenith_angles, [chapman_q_peak[angle] for angle in solar_zenith_angles], label='Chapman Theory')
    ax_peak_q.scatter(solar_zenith_angles, [q_values_max.get(angle, np.nan) for angle in solar_zenith_angles], label='Data')
    ax_peak_q.set_xlabel("Solar Zenith Angle (deg)")
    ax_peak_q.set_ylabel(r"Peak ionization rate (ions/m$^{3}$/s)")
    ax_peak_q.legend()
    pdf_pages.savefig(fig_peak_q)
    plt.close(fig_peak_q)
else:
    print("Cannot compare peak ionization rates as q_0 is not available.")

# --- Comparing Altitude of Peak Ionization ---
if 0 in q_values:
    peak_altitudes_data = {}
    for angle, q in q_values.items():
        peak_index = np.argmax(q)
        peak_altitudes_data[angle] = altitude[peak_index] / 1e3  # in km

    chapman_peak_altitudes = []
    scale_heights_used = []
    for angle in solar_zenith_angles:
        scale_height_m = scale_heights.get(angle, scale_heights.get(0, 25000)) # Get angle-specific H, default to H at 0 or 25000
        scale_height_km = scale_height_m / 1e3
        chapman_peak_altitude = z_m_chapman(angle, z_0_max_ref / 1e3, scale_height_km)
        chapman_peak_altitudes.append(chapman_peak_altitude)
        scale_heights_used.append(scale_height_km)

    label_string = "Chapman Theory"

    fig_peak_alt, ax_peak_alt = plt.subplots()
    ax_peak_alt.grid(True, linestyle=':')
    ax_peak_alt.scatter(solar_zenith_angles, chapman_peak_altitudes, label=label_string)
    ax_peak_alt.scatter(solar_zenith_angles, [peak_altitudes_data.get(angle, np.nan) for angle in solar_zenith_angles], label='Data')
    ax_peak_alt.set_xlabel("Solar Zenith Angle (deg)")
    ax_peak_alt.set_ylabel("Altitude of Peak Ionization (km)")
    ax_peak_alt.legend()
    pdf_pages.savefig(fig_peak_alt)
    plt.close(fig_peak_alt)
else:
    print("Cannot compare peak ionization altitudes as q_0 is not available.")

# --- Ionization Cross-sections Plot ---
wavelength_angstrom = df_2["Wavelength"] * 1e10  # Convert m to Ångstrøm
ion_cs_n2_cm2 = df_2["N2"] * 1e4  # Convert m^2 to cm^2
ion_cs_o_cm2 = df_2["O"] * 1e4
ion_cs_o2_cm2 = df_2["O2"] * 1e4

fig_cross_sections, ax_cross_sections = plt.subplots()
ax_cross_sections.grid(True, linestyle=':')
ax_cross_sections.plot(wavelength_angstrom, ion_cs_n2_cm2, label='N2')
ax_cross_sections.plot(wavelength_angstrom, ion_cs_o_cm2, label='O')
ax_cross_sections.plot(wavelength_angstrom, ion_cs_o2_cm2, label='O2')
ax_cross_sections.set_xlabel("Wavelength (Å)")
ax_cross_sections.set_ylabel("Ionization Cross Section (cm$^2$)")
ax_cross_sections.set_title("Ionization Cross Sections vs. Wavelength")
ax_cross_sections.set_yscale('log')
ax_cross_sections.legend()
pdf_pages.savefig(fig_cross_sections)
plt.close(fig_cross_sections)

# --- Finalize PDF ---
pdf_pages.close()

print("All plots have been saved to 'production_and_ionization_profiles.pdf'")