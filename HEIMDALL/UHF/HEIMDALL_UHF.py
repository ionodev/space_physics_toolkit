import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.colors as colors
from pyproj import Geod
import sys

# Radar location (Tromsø)
radar_lat_deg = 69.58
radar_lon_deg = 19.23
radar_alt_m = 86

# --- IMPORTANT: Set the correct file path here ---
file_path = 'beata-20250303-20250304.mat'
# -------------------------------------------------

# --- Dummy Data Generation (Remove this block if using a real file) ---
try:
    mat_data_test = scipy.io.loadmat(file_path)
    print(f"Successfully loaded data from '{file_path}'.")
    required_vars = ['ne', 'h', 'az', 'el', 'T', 'vi', 'Te', 'Ti']
    missing_vars = [var for var in required_vars if mat_data_test.get(var) is None]
    if missing_vars:
        print(f"Warning: The following variables are missing from '{file_path}': {', '.join(missing_vars)}")
    del mat_data_test
except FileNotFoundError:
    print(f"Warning: File '{file_path}' not found. Generating dummy data for demonstration.", file=sys.stderr)
    num_time_steps_dummy = 18 * 6
    num_altitudes_dummy = 50
    T_dummy = np.zeros((num_time_steps_dummy, 6))
    start_time = datetime(2025, 3, 3, 6, 0, 0)
    for i in range(num_time_steps_dummy):
        t_step = start_time + timedelta(minutes=i*10)
        T_dummy[i, :] = [t_step.year, t_step.month, t_step.day, t_step.hour, t_step.minute, t_step.second]
    h_dummy = np.linspace(100, 600, num_altitudes_dummy).reshape(1, -1)
    az_dummy = np.linspace(0, 360, num_time_steps_dummy)
    el_dummy = np.ones(num_time_steps_dummy) * 45
    ne_dummy = 10**(10 + 2 * np.random.rand(num_altitudes_dummy, num_time_steps_dummy))
    Te_dummy = 1000 + 2000 * np.random.rand(num_altitudes_dummy, num_time_steps_dummy)
    Ti_dummy = 500 + 1500 * np.random.rand(num_altitudes_dummy, num_time_steps_dummy)
    vi_dummy = (np.random.rand(num_altitudes_dummy, num_time_steps_dummy) - 0.5) * 1000
    mat_data = {
        'ne': ne_dummy, 'h': h_dummy, 'az': az_dummy, 'el': el_dummy, 'T': T_dummy,
        'Te': Te_dummy, 'Ti': Ti_dummy, 'vi': vi_dummy
    }
    using_dummy_data = True
except Exception as e:
    print(f"An error occurred while trying to load or generate dummy data: {e}", file=sys.stderr)
    sys.exit(1)
else:
    using_dummy_data = False
# --- End of Dummy Data Generation ---

# --- Main Data Loading and Processing ---
try:
    if not using_dummy_data:
        mat_data = scipy.io.loadmat(file_path)
    ne = mat_data.get('ne')
    h = mat_data.get('h')
    az = mat_data.get('az')
    el = mat_data.get('el')
    T = mat_data.get('T')
    vi = mat_data.get('vi')
    Te = mat_data.get('Te')
    Ti = mat_data.get('Ti')
    required_vars = {'ne': ne, 'h': h, 'az': az, 'el': el, 'T': T, 'vi': vi, 'Te': Te, 'Ti': Ti}
    missing_vars = [name for name, var in required_vars.items() if var is None]
    if missing_vars:
        raise ValueError(f"Could not load the following variables from the .mat file: {', '.join(missing_vars)}")
    if h.ndim > 1 and (h.shape[0] == 1 or h.shape[1] == 1): h = h.flatten()
    if az.ndim > 1 and (az.shape[0] == 1 or az.shape[1] == 1): az = az.flatten()
    if el.ndim > 1 and (el.shape[0] == 1 or el.shape[1] == 1): el = el.flatten()
    num_time_steps = T.shape[0]
    num_altitudes = h.shape[0]
    expected_shape = (num_altitudes, num_time_steps)
    for name, var_ref in [('ne', ne), ('Te', Te), ('Ti', Ti), ('vi', vi)]:
        var = locals()[name]
        if var.shape != expected_shape:
            print(f"Warning: Shape mismatch for '{name}'. Expected {expected_shape}, got {var.shape}. Trying to reshape.", file=sys.stderr)
            try:
                var = var.reshape(expected_shape)
                locals()[name] = var
                if name == 'ne': ne = var
                if name == 'Te': Te = var
                if name == 'Ti': Ti = var
                if name == 'vi': vi = var
                print(f"Successfully reshaped '{name}'.")
            except ValueError as reshape_err:
                raise ValueError(f"Could not reshape '{name}' from {var.shape} to {expected_shape}. Error: {reshape_err}")
    if az.shape[0] != num_time_steps or el.shape[0] != num_time_steps:
        raise ValueError(f"Azimuth ({az.shape}) or Elevation ({el.shape}) length does not match time steps ({num_time_steps})")
    latitudes = np.zeros_like(ne, dtype=float)
    geod = Geod(ellps='WGS84')
    for time_index in range(num_time_steps):
        azimuth_deg = az[time_index]
        elevation_deg = el[time_index]
        elevation_rad = np.deg2rad(elevation_deg)
        if elevation_deg <= 0 or np.isnan(elevation_deg):
            latitudes[:, time_index] = np.nan; continue
        for alt_index in range(num_altitudes):
            altitude_km = h[alt_index]
            if np.isnan(altitude_km):
                latitudes[alt_index, time_index] = np.nan; continue
            range_m = (altitude_km * 1000) / (np.sin(elevation_rad) + 1e-9)
            try:
                end_lon, end_lat, back_azimuth = geod.fwd(radar_lon_deg, radar_lat_deg, azimuth_deg, range_m)
                latitudes[alt_index, time_index] = end_lat
            except ValueError as geod_err:
                print(f"Warning: Geodetic calculation error at time index {time_index}, alt index {alt_index}: {geod_err}", file=sys.stderr)
                latitudes[alt_index, time_index] = np.nan
    print(f"Min calculated latitude: {np.nanmin(latitudes):.2f}, Max calculated latitude: {np.nanmax(latitudes):.2f}")
    time_objects = [datetime(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])) for row in T]
    print(f"The first time in the data is: {time_objects[0].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"The last time in the data is: {time_objects[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    hour_of_day = np.array([t.hour + t.minute / 60.0 + t.second / 3600.0 for t in time_objects])
    shifted_hour = (hour_of_day - 9) % 24
    angles = shifted_hour * (2 * np.pi / 24.0)
    lat_for_alt = np.nanmean(latitudes, axis=1)
    min_display_lat = 69
    radius = lat_for_alt - min_display_lat
    radius[radius < 0] = 0
    radius[np.isnan(radius)] = 0
    angle_grid, radius_grid = np.meshgrid(angles, radius)
    hours_to_label = np.arange(9, 33) % 24
    time_labels_all_hours = [f'{h:02d}' for h in hours_to_label]
    time_angles_all_hours = ((hours_to_label - 9) % 24) * (2 * np.pi / 24.0)
    latitude_ticks = np.arange(70, 84, 2)
    radius_ticks = latitude_ticks - min_display_lat
    valid_radius_ticks = radius_ticks[radius_ticks >= np.nanmin(radius)]
    valid_latitude_labels = [f'{lat}°N' for lat in latitude_ticks[radius_ticks >= np.nanmin(radius)]]
    if len(valid_radius_ticks) == 0:
        valid_radius_ticks = [np.nanmin(radius)]
        valid_latitude_labels = [f'{min_display_lat + np.nanmin(radius):.0f}°N']

    # --- Modified plotting function ---
    def create_polar_plot(ax, data, cmap, norm, title, cbar_label, cbar_ticks=None):
        masked_data = np.ma.masked_invalid(data)
        c = ax.pcolormesh(angle_grid, radius_grid, masked_data, cmap=cmap, norm=norm, shading='auto')
        cb = plt.colorbar(c, ax=ax, label=cbar_label, ticks=cbar_ticks, pad=0.1, shrink=0.8)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks(time_angles_all_hours)
        ax.set_xticklabels(time_labels_all_hours)
        ax.set_yticks(valid_radius_ticks)
        ax.set_yticklabels(valid_latitude_labels)
        min_r, max_r = np.nanmin(radius), np.nanmax(radius)
        ax.set_ylim(0, max_r * 1.05)
        ax.set_rlabel_position(90)
        ax.set_title(title, va='bottom', y=1.08)
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- Create a single figure with 2x2 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': 'polar'})
    axs = axs.flatten()  # Flatten to easily iterate over axes

    # --- Plot 1: Electron Density ---
    ne_vmin = np.nanpercentile(ne, 5)
    ne_vmax = np.nanpercentile(ne, 95)
    ne_norm = colors.LogNorm(vmin=ne_vmin, vmax=ne_vmax)
    ne_ticks = [1e10, 1e11, 1e12]
    create_polar_plot(axs[0], ne, 'jet', ne_norm, 'Electron Density (Latitude vs. UT)',
                      'Electron Density ($m^{-3}$)', ne_ticks)

    # --- Plot 2: Electron Temperature ---
    Te_vmin = np.nanpercentile(Te, 5)
    Te_vmax = np.nanpercentile(Te, 95)
    print(f"Te range (5th-95th percentile): {Te_vmin:.0f} K - {Te_vmax:.0f} K")
    Te_norm = colors.Normalize(vmin=max(0, Te_vmin), vmax=Te_vmax)
    create_polar_plot(axs[1], Te, 'jet', Te_norm, 'Electron Temperature (Latitude vs. UT)',
                      'Electron Temperature (K)')

    # --- Plot 3: Ion Temperature ---
    Ti_vmin = np.nanpercentile(Ti, 5)
    Ti_vmax = np.nanpercentile(Ti, 95)
    print(f"Ti range (5th-95th percentile): {Ti_vmin:.0f} K - {Ti_vmax:.0f} K")
    Ti_norm = colors.Normalize(vmin=max(0, Ti_vmin), vmax=Ti_vmax)
    create_polar_plot(axs[2], Ti, 'jet', Ti_norm, 'Ion Temperature (Latitude vs. UT)',
                      'Ion Temperature (K)')

    # --- Plot 4: Ion Drift77 Velocity ---
    vi_max_abs = np.nanpercentile(np.abs(vi), 98)
    vi_vmin = -vi_max_abs
    vi_vmax = vi_max_abs
    print(f"Vi range set to: {vi_vmin:.0f} m/s - {vi_vmax:.0f} m/s")
    vi_norm = colors.Normalize(vmin=vi_vmin, vmax=vi_vmax)
    create_polar_plot(axs[3], vi, 'bwr', vi_norm, 'Ion Drift Velocity (Latitude vs. UT)',
                      'Ion Drift Velocity ($m/s$)')

    # --- Adjust layout and save as PNG ---
    plt.tight_layout()
    plt.savefig('plots_combined_UHF.png', format='png', bbox_inches='tight')
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Cannot proceed without data.", file=sys.stderr)
except ValueError as e:
    print(f"Data Error: {e}", file=sys.stderr)
except ImportError as e:
    print(f"Import Error: {e}. Please ensure required libraries (scipy, numpy, matplotlib, pyproj) are installed.", file=sys.stderr)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()