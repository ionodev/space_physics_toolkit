import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection
import scipy.io
# Import necessary library for CCF and ACF
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf  # Import ACF plotting function
import pandas as pd  # Import pandas for easy NaN handling
import warnings

# Suppress specific warnings if they appear during interpolation/differencing with NaNs
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning) # Re-enable if needed


# --- STEP 1: Data Loading and Basic Preparation ---

# Define the time range
start_time = datetime(2025, 3, 3, 9, 0, 0)  # 9 UT March 3, 2025
end_time = datetime(2025, 3, 4, 9, 0, 0)  # 9 UT March 4, 2025

# List to store the number of data points for each dataset within the time range
point_counts = []
print("--- Loading Data and Counting Points ---")

# Load ACE magnetometer data (Bz component) for both days
ace_files = ['20250303_ace_mag_1m.txt', '20250304_ace_mag_1m.txt']
ace_datetimes_raw = []
ace_bz_raw = []
ace_status_raw = []

for ace_file in ace_files:
    try:
        data = np.loadtxt(ace_file, skiprows=14, usecols=(0, 1, 2, 3, 5, 6, 9), dtype=str)
        for row in data:
            dt = datetime.strptime(f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]}", "%Y %m %d %H%M")
            if start_time <= dt <= end_time:
                ace_datetimes_raw.append(dt)
                ace_status_raw.append(int(row[5]))
                ace_bz_raw.append(float(row[6]))
    except FileNotFoundError:
        print(f"Warning: ACE file {ace_file} not found. Skipping.")
    except Exception as e:
        print(f"Warning: Error reading {ace_file}: {e}. Skipping.")

ace_datetimes = np.array(ace_datetimes_raw)
ace_status = np.array(ace_status_raw)
ace_bz = np.array(ace_bz_raw)
if len(ace_datetimes) > 0:
    valid_ace = (ace_status == 0) & (ace_bz != -999.9)
    ace_datetimes = ace_datetimes[valid_ace]
    ace_bz = ace_bz[valid_ace]
    point_counts.append(len(ace_datetimes))
    print(f"ACE Bz: Found {len(ace_datetimes)} valid points in range.")
else:
    print("ACE Bz: No data found in time range.")
    ace_datetimes = np.array([])  # Ensure it's an empty array
    ace_bz = np.array([])

# Calculate RAW hours for ACE Bz (used for CCF) if data exists
ace_hours_raw = np.array([(dt - start_time).total_seconds() / 3600.0 for dt in ace_datetimes]) if len(
    ace_datetimes) > 0 else np.array([])

# Calculate APPROXIMATE PHYSICALLY DELAYED hours for ACE Bz (used for final plot only)
Bz_delay_minutes_plot = 0  # 45 # Your visual alignment delay
Bz_delay_plot = Bz_delay_minutes_plot / 60.0
ace_hours_plot = ace_hours_raw + Bz_delay_plot  # Add delay only for the plot version

# REMOVED: NAL Magnetometer Loading Section

# Load magnetometer data (horizontal component) for TSO
tso_files = ['TSO.txt', 'TSO_2.txt']
tso_datetimes_raw = []
tso_mag_horizontal_raw = []
for tso_file in tso_files:
    try:
        data = np.loadtxt(tso_file, skiprows=6, usecols=(0, 1, 3), dtype=str)
        for row in data:
            dt = datetime.strptime(f"{row[0]} {row[1]}", "%d/%m/%Y %H:%M:%S")
            if start_time <= dt <= end_time:
                tso_datetimes_raw.append(dt)
                tso_mag_horizontal_raw.append(float(row[2]))
    except FileNotFoundError:
        print(f"Warning: TSO Mag file {tso_file} not found. Skipping.")
    except Exception as e:
        print(f"Warning: Error reading {tso_file}: {e}. Skipping.")

tso_datetimes = np.array(tso_datetimes_raw)
tso_mag_horizontal = np.array(tso_mag_horizontal_raw)
if len(tso_datetimes) > 0:
    point_counts.append(len(tso_datetimes))
    print(f"TSO Mag: Found {len(tso_datetimes)} points in range.")
else:
    print("TSO Mag: No data found in time range.")
    tso_datetimes = np.array([])
    tso_mag_horizontal = np.array([])

tso_mag_hours_raw = np.array([(dt - start_time).total_seconds() / 3600.0 for dt in tso_datetimes]) if len(
    tso_datetimes) > 0 else np.array([])  # RAW hours (for CCF)
tso_mag_delay_plot = 0  # 0.40 # Your visual alignment delay
tso_mag_hours_plot = tso_mag_hours_raw + tso_mag_delay_plot  # Hours with visual delay (for final plot)

# Load EISCAT VHF electron density data
file_path_vhf = "bella-20250303-20250304.mat"
ne_vhf = None;
h_vhf = None;
T_vhf = None  # Initialize
eiscat_times_vhf = np.array([])  # Initialize
eiscat_hours_vhf = np.array([])  # Initialize
median_ne_vhf = np.array([])  # Initialize
rolling_median_vhf = np.array([])  # Initialize
try:
    mat_data_vhf = scipy.io.loadmat(file_path_vhf)
    ne_vhf = mat_data_vhf.get('ne');
    h_vhf = mat_data_vhf.get('h');
    T_vhf = mat_data_vhf.get('T')

    # Convert EISCAT VHF time array to datetime and filter
    eiscat_times_vhf_raw = [];
    eiscat_indices_vhf = []
    for i, t_vec in enumerate(T_vhf):
        year, month, day, hour, minute, second = map(int, t_vec)
        dt = datetime(year, month, day, hour, minute, second)
        if start_time <= dt <= end_time: eiscat_times_vhf_raw.append(dt); eiscat_indices_vhf.append(i)
    eiscat_times_vhf = np.array(eiscat_times_vhf_raw);
    eiscat_indices_vhf = np.array(eiscat_indices_vhf)
    if len(eiscat_times_vhf) > 0:
        point_counts.append(len(eiscat_times_vhf))
        print(f"EISCAT VHF: Found {len(eiscat_times_vhf)} points in range.")
        eiscat_hours_vhf = np.array(
            [(t - start_time).total_seconds() / 3600.0 for t in eiscat_times_vhf])  # RAW hours (for CCF and plot)

        # Prepare EISCAT VHF data (E-region: 90-150 km)
        h_vhf = h_vhf.flatten();
        mask_vhf = (h_vhf >= 90) & (h_vhf <= 150)
        h_filtered_vhf = h_vhf[mask_vhf]
        ne_filtered_vhf = ne_vhf[mask_vhf, :][:, eiscat_indices_vhf]
        ne_filtered_vhf = np.where(ne_filtered_vhf > 0, ne_filtered_vhf, np.nan);
        ne_filtered_vhf = np.where(np.isfinite(ne_filtered_vhf), ne_filtered_vhf, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning);
            median_ne_vhf = np.nanmedian(ne_filtered_vhf,
                                         axis=0)

        # Compute rolling median for VHF
        window_size = 15
        rolling_median_vhf = np.full_like(median_ne_vhf, np.nan)
        if len(median_ne_vhf) >= window_size:
            for i in range(window_size - 1, len(median_ne_vhf)):
                window = median_ne_vhf[i - window_size + 1:i + 1]
                if not np.all(np.isnan(window)):
                    with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning);
                    rolling_median_vhf[i] = np.nanmedian(window)
        else:
            rolling_median_vhf = median_ne_vhf  # Not enough points for rolling median
    else:
        print("EISCAT VHF: No data found in time range.")

except FileNotFoundError:
    print(f"Warning: EISCAT VHF file {file_path_vhf} not found.")
except Exception as e:
    print(f"Warning: Error processing EISCAT VHF data: {e}")

# Load EISCAT UHF electron density data
file_path_uhf = "beata-20250303-20250304.mat"
ne_uhf = None;
h_uhf = None;
T_uhf = None
eiscat_times_uhf = np.array([])
eiscat_hours_uhf = np.array([])
median_ne_uhf = np.array([])
rolling_median_uhf = np.array([])
try:
    mat_data_uhf = scipy.io.loadmat(file_path_uhf)
    ne_uhf = mat_data_uhf.get('ne');
    h_uhf = mat_data_uhf.get('h');
    T_uhf = mat_data_uhf.get('T')

    # Convert EISCAT UHF time array to datetime and filter
    eiscat_times_uhf_raw = [];
    eiscat_indices_uhf = []
    for i, t_vec in enumerate(T_uhf):
        year, month, day, hour, minute, second = map(int, t_vec)
        dt = datetime(year, month, day, hour, minute, second)
        if start_time <= dt <= end_time: eiscat_times_uhf_raw.append(dt); eiscat_indices_uhf.append(i)
    eiscat_times_uhf = np.array(eiscat_times_uhf_raw);
    eiscat_indices_uhf = np.array(eiscat_indices_uhf)
    if len(eiscat_times_uhf) > 0:
        point_counts.append(len(eiscat_times_uhf))
        print(f"EISCAT UHF: Found {len(eiscat_times_uhf)} points in range.")
        eiscat_hours_uhf = np.array(
            [(t - start_time).total_seconds() / 3600.0 for t in eiscat_times_uhf])  # RAW hours (for CCF and plot)

        # Prepare EISCAT UHF data (E-region: 90-150 km)
        h_uhf = h_uhf.flatten();
        mask_uhf = (h_uhf >= 90) & (h_uhf <= 150)
        h_filtered_uhf = h_uhf[mask_uhf]
        ne_filtered_uhf = ne_uhf[mask_uhf, :][:, eiscat_indices_uhf]
        ne_filtered_uhf = np.where(ne_filtered_uhf > 0, ne_filtered_uhf, np.nan);
        ne_filtered_uhf = np.where(np.isfinite(ne_filtered_uhf), ne_filtered_uhf, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning);
            median_ne_uhf = np.nanmedian(ne_filtered_uhf,
                                         axis=0)

        # Compute rolling median for UHF
        window_size = 15  # Same as VHF
        rolling_median_uhf = np.full_like(median_ne_uhf, np.nan)
        if len(median_ne_uhf) >= window_size:
            for i in range(window_size - 1, len(median_ne_uhf)):
                window = median_ne_uhf[i - window_size + 1:i + 1]
                if not np.all(np.isnan(window)):
                    with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning);
                    rolling_median_uhf[i] = np.nanmedian(window)
        else:
            rolling_median_uhf = median_ne_uhf  # Not enough points for rolling median
    else:
        print("EISCAT UHF: No data found in time range.")

except FileNotFoundError:
    print(f"Warning: EISCAT UHF file {file_path_uhf} not found.")
except Exception as e:
    print(f"Warning: Error processing EISCAT UHF data: {e}")

# Riometer instruments list (Keep only SKN_30)
riometer_instruments = [
    ('SKN_30', ['SKN_30.txt', 'SKN_30_2.txt'])
]

# Dictionary to store loaded riometer data BEFORE interpolation
riometer_raw_data = {}

# Load Riometer data and count points
for instrument, files in riometer_instruments:
    timestamps_sec_raw = [];
    values_raw = []
    for file in files:
        try:
            # Handle potential deprecation warning for utcfromtimestamp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                data = np.loadtxt(file, usecols=(0, 1))
                file_timestamps_unix = data[:, 0]
                file_values = data[:, 1]
                file_datetimes = [datetime.utcfromtimestamp(ts) for ts in file_timestamps_unix]

            for dt, val in zip(file_datetimes, file_values):
                if start_time <= dt <= end_time:
                    timestamps_sec_raw.append((dt - start_time).total_seconds())
                    values_raw.append(val)
        except FileNotFoundError:
            print(f"Warning: Riometer file {file} for {instrument} not found. Skipping.")
        except Exception as e:
            print(f"Warning: Error reading {file} for {instrument}: {e}. Skipping.")

    if len(timestamps_sec_raw) > 0:
        timestamps_sec = np.array(timestamps_sec_raw);
        values = np.array(values_raw)
        # Sort data by time before storing
        sorted_indices = np.argsort(timestamps_sec)
        timestamps_sec_sorted = timestamps_sec[sorted_indices]
        values_sorted = values[sorted_indices]
        # Store raw (but sorted) data for later interpolation
        riometer_raw_data[instrument] = {'hours': timestamps_sec_sorted / 3600.0, 'values': values_sorted}
        point_counts.append(len(timestamps_sec_sorted))
        print(f"Riometer {instrument}: Found {len(timestamps_sec_sorted)} points in range.")
    else:
        print(f"Riometer {instrument}: No data found in time range.")
        riometer_raw_data[instrument] = {'hours': np.array([]), 'values': np.array([])}

# --- Determine num and Create Final Time Grid ---
if not point_counts:  # Check if any data was loaded
    print("Error: No data points found for any instrument in the specified time range.")
    print("Setting num to default value 1000, but results might be unreliable.")
    num = 1000  # Default value if no data
else:
    num = max(point_counts)
    print(f"\nDetermined 'num' (max points in any dataset): {num}")

# Check if num is sensible (e.g., not extremely small if data exists)
if num < 10 and any(pc > 0 for pc in point_counts):
    print(f"Warning: Calculated 'num' ({num}) is very small. Check data loading and time ranges.")
elif num <= 1:  # np.linspace needs at least 2 points for a range
    print(f"Warning: Calculated 'num' ({num}) is too small for linspace. Setting num=2.")
    num = 2

hours_common_final = np.linspace(0, 24, num)  # The final common time grid in hours from start_time
time_step_hours_final = hours_common_final[1] - hours_common_final[0] if len(hours_common_final) > 1 else 0
print(f"Final time grid: {num} points over 24 hours. Resolution: {time_step_hours_final * 60:.2f} minutes")

# --- MEDIAN FILTERING AND INTERPOLATION ONTO FINAL GRID ---
print("\n--- Preparing final data grid for plot & analysis ---")


# Function to compute median every n samples (moved here, used after interpolation)
def median_every_n(data, n):
    n = int(n);
    R = RuntimeWarning
    if n <= 0: return data
    data = np.asarray(data)  # Ensure it's a numpy array
    if len(data) < n:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=R)
            return np.array([np.nanmedian(data)]) if len(data) > 0 else np.array([])
    # Ensure sufficient points for reshaping
    num_windows = len(data) // n
    if num_windows == 0:
        return np.array([])  # Not enough data for even one full window
    trimmed_len = num_windows * n
    trimmed_data = data[:trimmed_len]
    reshaped_data = trimmed_data.reshape(-1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=R)
        medians = np.nanmedian(reshaped_data, axis=1)
    return medians


# Set the window size for the final median filter (applied AFTER interpolation)
final_median_window_minutes = 1.0
if time_step_hours_final > 0:
    # Ensure n_median_final is at least 1
    n_median_final = max(1, int(round(final_median_window_minutes / (time_step_hours_final * 60.0))))
else:
    n_median_final = 1  # Avoid division by zero if time step is 0
print(f"Final median filter window: {n_median_final} points (approx {final_median_window_minutes:.1f} min)")

# Interpolate Riometer data onto the final grid and apply median filter
riometer_data_final = []
riometer_labels_final = []
for instrument, files in riometer_instruments:  # Use the original list to maintain order (now only SKN_30)
    riometer_labels_final.append(instrument)
    raw_data = riometer_raw_data[instrument]
    if len(raw_data['hours']) > 1:  # Need at least 2 points to interpolate
        interp_func = interp1d(raw_data['hours'], raw_data['values'], kind='linear', bounds_error=False,
                               fill_value=np.nan)
        values_interp_final = interp_func(hours_common_final)
        values_median_final = median_every_n(values_interp_final, n_median_final)
        riometer_data_final.append(values_median_final)
    else:
        print(
            f"Skipping interpolation for {instrument} due to insufficient points ({len(raw_data['hours'])}). Filling with NaNs.")
        # Calculate expected length after median filtering a dummy array
        dummy_median_len = len(median_every_n(np.zeros(len(hours_common_final)), n_median_final))
        riometer_data_final.append(np.full(dummy_median_len, np.nan))

# Determine num_points_final based on the actual length *after* median filtering
if riometer_data_final and len(riometer_data_final[0]) > 0:
    num_points_final = len(riometer_data_final[0])  # Length after potential median filtering
    # Correct the length of any failed interpolations if they exist (though unlikely with only one instrument now)
    for i in range(len(riometer_data_final)):
        if len(riometer_data_final[i]) != num_points_final:
            riometer_data_final[i] = np.full(num_points_final, np.nan)
else:
    # Fallback if no riometer data was successfully processed
    num_points_final = len(median_every_n(np.zeros(len(hours_common_final)), n_median_final))
    print(f"Warning: No valid riometer data after filtering. Using fallback length: {num_points_final}")
    # Ensure riometer_data_final list elements (if any) have the correct length
    for i in range(len(riometer_data_final)):
        riometer_data_final[i] = np.full(num_points_final, np.nan)
    # If the list itself is empty, add a placeholder if needed downstream
    if not riometer_data_final:
        riometer_data_final.append(np.full(num_points_final, np.nan))

hours_final = np.linspace(0, 24, num_points_final)  # Time axis for final plot AND analysis
time_step_hours_final = hours_final[1] - hours_final[0] if len(hours_final) > 1 else 0  # Recalculate final time step
print(
    f"Time grid after median filter: {num_points_final} points, New Resolution: {time_step_hours_final * 60:.2f} minutes")

# --- Interpolate other data FOR CCF ANALYSIS (using RAW times, no manual delays) ---
print("Interpolating other data for CCF Analysis (NO manual delays)...")


def interpolate_safe(x_known, y_known, x_new_target_axis, label):
    """Helper function for safe interpolation and median filtering."""
    target_len = len(x_new_target_axis)
    # Determine the common grid length before median filtering
    common_grid_len = num  # From the initial max point count

    if len(x_known) > 1 and len(y_known) == len(x_known):
        try:
            # Interpolate onto the high-resolution common grid first
            x_common_high_res = np.linspace(0, 24, common_grid_len)
            interp_func = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value=np.nan)
            y_interp_common = interp_func(x_common_high_res)

            # Apply final median filter
            y_median = median_every_n(y_interp_common, n_median_final)

            # Ensure output length matches the target axis length
            if len(y_median) != target_len:
                print(
                    f"Warning: Length mismatch for {label} after median filter ({len(y_median)}) vs target ({target_len}). Re-interpolating median data.")
                # Need a time axis for the median data to re-interpolate
                # Assume median filter reduces length proportionally
                if len(y_interp_common) > 0 and n_median_final > 0 and len(y_median) > 0:  # check y_median has data
                    median_hours = np.linspace(0, 24, len(y_median))  # Approximate time axis for median data
                    if len(median_hours) > 1 and len(y_median) == len(median_hours):
                        # Interpolate the median data onto the final target time axis
                        interp_median_func = interp1d(median_hours, y_median, kind='linear', bounds_error=False,
                                                      fill_value=np.nan)
                        y_final = interp_median_func(x_new_target_axis)
                        return y_final
                    else:
                        print(
                            f"  Cannot re-interpolate median data for {label} (median_hours len <=1 or length mismatch). Returning NaNs.")
                        return np.full(target_len, np.nan)
                else:
                    print(
                        f"  Cannot determine median time axis for {label} (interp_common or y_median empty). Returning NaNs.")
                    return np.full(target_len, np.nan)
            else:
                return y_median

        except ValueError as e:
            print(f"Interpolation error for {label}: {e}. Returning NaNs.")
            return np.full(target_len, np.nan)
    else:
        print(
            f"Skipping interpolation for {label}: Insufficient points ({len(x_known)}) or mismatched lengths. Returning NaNs.")
        return np.full(target_len, np.nan)


# REMOVED: nal_mag_ccf calculation
tso_mag_ccf = interpolate_safe(tso_mag_hours_raw, tso_mag_horizontal, hours_final, "TSO Mag (CCF)")
vhf_ne_ccf = interpolate_safe(eiscat_hours_vhf, rolling_median_vhf, hours_final,
                              "VHF Ne (CCF)")  # Using rolling median as input
uhf_ne_ccf = interpolate_safe(eiscat_hours_uhf, rolling_median_uhf, hours_final,
                              "UHF Ne (CCF)")  # Using rolling median as input
bz_ccf = interpolate_safe(ace_hours_raw, ace_bz, hours_final, "ACE Bz (CCF)")

# Riometer data for CCF is the already interpolated+median filtered data
skn_30_ccf = riometer_data_final[0] if len(riometer_data_final) > 0 else np.full(num_points_final, np.nan)

# --- Interpolate data FOR FINAL PLOT (using times WITH manual/visual delays) ---
print("Interpolating data for Final Plot (WITH manual delays)...")
print(f"Using fixed ACE Bz delay FOR PLOT: {Bz_delay_minutes_plot:.1f} minutes ({Bz_delay_plot:.2f} hours)")
# REMOVED: NAL Mag delay print
print(f"Using fixed TSO Mag delay FOR PLOT: {tso_mag_delay_plot * 60:.1f} minutes ({tso_mag_delay_plot:.2f} hours)")

# Use the safe interpolation function again, now with _plot hours
tso_mag_plot = interpolate_safe(tso_mag_hours_plot, tso_mag_horizontal, hours_final, "TSO Mag (Plot)")
bz_plot = interpolate_safe(ace_hours_plot, ace_bz, hours_final, "ACE Bz (Plot)")

# EISCAT uses raw times for both plot and analysis - already interpolated as vhf_ne_ccf / uhf_ne_ccf
vhf_ne_plot = vhf_ne_ccf
uhf_ne_plot = uhf_ne_ccf

# --- STEP 2: Difference the Time Series (Using the UNDELAYED CCF data) ---
print("\n--- Differencing the UNDELAYED Time Series for CCF ---")


def safe_diff(series):
    # Ensure series is numpy array
    series = np.asarray(series)
    if len(series) < 2:  # Cannot difference if less than 2 points
        return np.array([])
    if np.all(np.isnan(series)):
        # Return NaNs matching the expected output length (len-1)
        return np.full(max(0, len(series) - 1), np.nan)
    else:
        # Difference, maintaining NaN propagation where possible
        diff_series = np.full(len(series) - 1, np.nan)
        valid_indices = np.where(np.isfinite(series[:-1]) & np.isfinite(series[1:]))[0]
        diff_series[valid_indices] = series[valid_indices + 1] - series[valid_indices]
        return diff_series


# Use the _ccf versions (which are now post-median filter)
diff_skn_30 = safe_diff(skn_30_ccf)
diff_tso_mag = safe_diff(tso_mag_ccf)
diff_vhf_ne = safe_diff(vhf_ne_ccf)
diff_uhf_ne = safe_diff(uhf_ne_ccf)
diff_bz = safe_diff(bz_ccf)  # Use undelayed Bz difference

# Time axis for differenced data (N-1 points)
hours_diff_final = hours_final[1:] if len(hours_final) > 1 else np.array([])

# --- PLOT DIFFERENCED TIME SERIES (Reduced Panels) ---
print("\n--- Plotting Differenced Time Series used for CCF (No Manual Delays - Reduced) ---")

if len(hours_diff_final) > 0:  # Only plot if differencing was possible
    # Reduced number of rows from 8 to 5
    fig_diff, axes_diff = plt.subplots(nrows=5, ncols=1, figsize=(12, 2.5 * 5), sharex=True)
    time_res_label_final = f"(Δ per {time_step_hours_final * 60:.1f} min)" if time_step_hours_final > 0 else "(Δ)"

    # Plot Diff(SKN_30) - Now at index 0
    axes_diff[0].plot(hours_diff_final, diff_skn_30, 'b-',
                      label=f'Diff(Cosmic_noise)')  # Index 0 for SKN_30
    axes_diff[0].set_ylabel(f'ΔNoise (arb. units)');
    axes_diff[0].grid(True);
    axes_diff[0].legend(loc='upper right')
    axes_diff[0].set_title('Differenced Time Series')

    # Plot Diff(TSO Mag H - Raw Time) - Now at index 1
    axes_diff[1].plot(hours_diff_final, diff_tso_mag, 'r-', label='Diff(H_component)')
    axes_diff[1].set_ylabel(f'ΔH (nT)');
    axes_diff[1].grid(True);
    axes_diff[1].legend(loc='upper right')

    # Plot Diff(VHF Ne) - Now at index 2
    axes_diff[2].plot(hours_diff_final, diff_vhf_ne, 'b-', label='Diff(VHF_Ne)')
    axes_diff[2].set_ylabel(f'ΔNe (m^-3)');
    axes_diff[2].grid(True);
    axes_diff[2].legend(loc='upper right')

    # Plot Diff(UHF Ne) - Now at index 3
    axes_diff[3].plot(hours_diff_final, diff_uhf_ne, 'b-', label='Diff(UHF_Ne)')
    axes_diff[3].set_ylabel(f'ΔNe (m^-3)');
    axes_diff[3].grid(True);
    axes_diff[3].legend(loc='upper right')

    # Plot Diff(ACE Bz - Raw Time) - Now at index 4
    axes_diff[4].plot(hours_diff_final, diff_bz, 'k-', label='Diff(Bz)')
    axes_diff[4].axhline(0, color='gray', linestyle=':', linewidth=0.5)
    axes_diff[4].set_ylabel(f'ΔBz (nT)');
    axes_diff[4].grid(True);
    axes_diff[4].legend(loc='upper right')

    # --- FIXED X-AXIS SETUP for Differenced Plot ---
    axes_diff[4].set_xlabel('Time (UT)')  # Set label to UT (Now on last axis index 4)
    axes_diff[4].set_xticks(np.arange(0, 25, 1))  # Set ticks every 1 hour on the 0-24 scale
    tick_labels = [str((h + 9) % 24) for h in range(25)]  # Calculate UT labels (9 to 8)
    axes_diff[4].set_xticklabels(tick_labels)  # Apply the UT labels
    axes_diff[4].set_xlim(0, 24)  # Keep limit from 0 to 24 relative hours
    # --- End of Fixed X-Axis Setup ---

    plt.tight_layout(rect=[0, 0, 1, 0.97])


    # Add format_coord for interactivity
    def format_coord_diff_builder(series, series_label, time_axis):
        # Ensure series and time_axis are numpy arrays for efficient indexing
        series_np = np.asarray(series)
        time_axis_np = np.asarray(time_axis)

        def format_coord(x, y):
            # Find UT hour corresponding to x (hours since start)
            ut_hour = (x + 9) % 24
            # Find closest index for data value display
            if len(time_axis_np) > 0:
                # Ensure x is within the bounds of the time axis for finding index
                if x < time_axis_np[0] or x > time_axis_np[-1]:
                    val_str = "N/A (out of time bounds)"
                else:
                    idx = np.argmin(np.abs(time_axis_np - x));
                    if idx < len(series_np):
                        actual_value = series_np[idx]
                        # Format value carefully
                        if np.isnan(actual_value):
                            val_str = "NaN"
                        elif abs(actual_value) > 1e4 or (abs(actual_value) > 0 and abs(actual_value) < 1e-2):
                            val_str = f'{actual_value:.2e}'
                        else:
                            val_str = f'{actual_value:.2f}'
                    else:
                        val_str = "N/A (index out of bounds)"
            else:
                val_str = "N/A (no time axis)"
            # Display UT hour and value
            return f'Time: {ut_hour:.1f} UT ({x:.2f} hrs), Value ({series_label}): {val_str}'

        return format_coord


    if len(hours_diff_final) > 0:  # Add formatters only if there's data
        axes_diff[0].format_coord = format_coord_diff_builder(diff_skn_30, f'Diff({riometer_labels_final[0]})',
                                                              hours_diff_final)  # Now index 0
        axes_diff[1].format_coord = format_coord_diff_builder(diff_tso_mag, 'Diff(TSO H)',
                                                              hours_diff_final)  # Now index 1
        axes_diff[2].format_coord = format_coord_diff_builder(diff_vhf_ne, 'Diff(VHF Ne)',
                                                              hours_diff_final)  # Now index 2
        axes_diff[3].format_coord = format_coord_diff_builder(diff_uhf_ne, 'Diff(UHF Ne)',
                                                              hours_diff_final)  # Now index 3
        axes_diff[4].format_coord = format_coord_diff_builder(diff_bz, 'Diff(ACE Bz)', hours_diff_final)  # Now index 4
    # plt.savefig('Fig_1.png')
    plt.show()
    # plt.close(fig_diff)
else:
    print("Skipping differenced plot: Not enough data points after interpolation/filtering.")


# --- STEP 3 & 4: CCF Calculation and Combined Plotting (TSO vs Others) ---

# Helper function (robust version kept for NaN handling within CCF)
def calculate_ccf_robust(x, y, max_lags):
    x = np.asarray(x)  # Ensure numpy arrays
    y = np.asarray(y)
    lags_out = np.arange(-max_lags, max_lags + 1)
    corrs_out = np.full(len(lags_out), np.nan)
    n_valid_out = 0

    if len(x) != len(y): print(
        f"Error: Input series lengths differ: {len(x)} vs {len(y)}"); return lags_out, corrs_out, n_valid_out
    if len(x) < max_lags + 2:
        print(
            f"Warning: Series length ({len(x)}) too short for max_lags ({max_lags}). Needs > {max_lags + 1}. Returning NaNs.")
        return lags_out, corrs_out, n_valid_out

    valid_idx = np.where(np.isfinite(x) & np.isfinite(y))[0]
    min_overlap_needed = max(max_lags + 1, 10)
    if len(valid_idx) < min_overlap_needed:
        print(
            f"Warning: Insufficient valid overlapping points ({len(valid_idx)}) for lags ({max_lags}). Needs >= {min_overlap_needed}. Returning NaNs.")
        return lags_out, corrs_out, n_valid_out

    x_valid = x[valid_idx];
    y_valid = y[valid_idx];
    n_valid = len(x_valid)

    if n_valid > 1 and (np.std(x_valid) < 1e-10 or np.std(y_valid) < 1e-10):
        print(
            "Warning: One or both series have near-zero standard deviation after NaN removal. CCF is undefined. Returning NaNs.")
        return lags_out, corrs_out, n_valid
    elif n_valid <= 1:
        print(
            f"Warning: Only {n_valid} valid overlapping point(s). CCF is undefined. Returning NaNs.")
        return lags_out, corrs_out, n_valid

    try:
        if len(x_valid) <= max_lags or len(y_valid) <= max_lags:
            print(
                f"Warning: Valid series length ({len(x_valid)}) too short for ccf with max_lags ({max_lags}). Returning NaNs.")
            return lags_out, corrs_out, n_valid

        ccf_pos = ccf(x_valid, y_valid, adjusted=False, fft=True);
        ccf_neg = ccf(y_valid, x_valid, adjusted=False, fft=True)
        if len(ccf_pos) <= max_lags or len(ccf_neg) <= max_lags:
            print(f"Warning: CCF result length too short for max_lags ({max_lags}). Returning NaNs.")
            return lags_out, corrs_out, n_valid

        corr_pos = ccf_pos[0:max_lags + 1];
        corr_neg = ccf_neg[1:max_lags + 1][::-1]
        corrs = np.concatenate((corr_neg, corr_pos));
        lags = np.arange(-max_lags, max_lags + 1)
        if np.all(np.isnan(corrs)):
            print("Warning: CCF calculation resulted in all NaNs. Check input data.")
        return lags, corrs, n_valid
    except Exception as e:
        print(
            f"Error during CCF calculation for series lengths {len(x_valid)}, {len(y_valid)} with max_lags {max_lags}: {e}. Returning NaNs.")
        return lags_out, corrs_out, n_valid


# --- Define pairs and parameters (TSO vs Others) ---
ccf_pairs_info = [
    {'label1': 'TSO_Mag', 'label2': riometer_labels_final[0], 'series1_diff': diff_tso_mag, 'series2_diff': diff_skn_30,
     'series1_raw': tso_mag_ccf, 'series2_raw': skn_30_ccf},
    {'label1': 'TSO_Mag', 'label2': 'VHF_Ne', 'series1_diff': diff_tso_mag, 'series2_diff': diff_vhf_ne,
     'series1_raw': tso_mag_ccf, 'series2_raw': vhf_ne_ccf},
    {'label1': 'TSO_Mag', 'label2': 'UHF_Ne', 'series1_diff': diff_tso_mag, 'series2_diff': diff_uhf_ne,
     'series1_raw': tso_mag_ccf, 'series2_raw': uhf_ne_ccf},
    {'label1': 'TSO_Mag', 'label2': 'Bz', 'series1_diff': diff_tso_mag, 'series2_diff': diff_bz,
     'series1_raw': tso_mag_ccf, 'series2_raw': bz_ccf}
]

max_lag_minutes_ccf_plot = 60  # For CCF plot display range +/- 60 minutes

if time_step_hours_final > 0:
    lag_points_per_minute_final = 1.0 / (time_step_hours_final * 60.0)
    max_lag_points_calc = int(
        round(max_lag_minutes_ccf_plot * lag_points_per_minute_final))  # Lags to CALCULATE for CCF

    if num_points_final <= 1:
        max_lag_points_calc = 0
    elif max_lag_points_calc >= num_points_final - 1:
        print(
            f"Warning: Calculated max_lag_points_calc ({max_lag_points_calc}) is too large for series length ({num_points_final}). Adjusting.")
        max_lag_points_calc = max(0, (num_points_final - 2) // 2 if num_points_final > 2 else 0)  # Ensure it's safe
        # Update max_lag_minutes_ccf_plot if max_lag_points_calc was drastically reduced
        # This ensures the plot x-axis matches the calculated range if it was capped.
        if lag_points_per_minute_final > 0:
            max_lag_minutes_ccf_plot = max_lag_points_calc / lag_points_per_minute_final
        else:  # Should not happen if time_step_hours_final > 0
            max_lag_minutes_ccf_plot = 0

    if max_lag_points_calc < 1 and num_points_final > 2:
        print(f"Warning: Calculated max_lag_points_calc ({max_lag_points_calc}) is less than 1. Setting to 1.")
        max_lag_points_calc = 1
        if lag_points_per_minute_final > 0:
            max_lag_minutes_ccf_plot = max_lag_points_calc / lag_points_per_minute_final
        else:
            max_lag_minutes_ccf_plot = 0


elif num_points_final <= 1:  # If time_step is 0 and no points
    print("Warning: Final time step is zero or no points. Setting max_lag_points_calc to 0.")
    lag_points_per_minute_final = 1.0
    max_lag_points_calc = 0
    max_lag_minutes_ccf_plot = 0
else:  # time_step_hours_final is 0 but num_points_final > 1 (should ideally not happen with linspace)
    print(
        "Warning: Final time step is zero. Cannot calculate lag points reliably. Setting max_lag_points_calc to a small default (e.g., 10 if data allows).")
    lag_points_per_minute_final = 1.0
    max_lag_points_calc = min(10, (num_points_final - 2) // 2 if num_points_final > 2 else 0)
    if lag_points_per_minute_final > 0:
        max_lag_minutes_ccf_plot = max_lag_points_calc / lag_points_per_minute_final
    else:
        max_lag_minutes_ccf_plot = 0

print(
    f"Calculating CCF up to {max_lag_points_calc} lag points (for approx +/-{max_lag_minutes_ccf_plot:.1f} minutes display)")

# --- Combined Plot for DIFFERENCED CCF (TSO vs Others) ---
print("\n--- Calculating and Plotting Combined CCF (DIFFERENCED Data - TSO vs Others) ---")
fig_ccf_diff, axes_ccf_diff = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes_ccf_diff_flat = axes_ccf_diff.flatten()

display_label_map = {
    'TSO_Mag': 'H_component',
    'VHF_Ne': 'VHF_Ne',
    'UHF_Ne': 'UHF_Ne',
    'Bz': 'Bz',
    riometer_labels_final[0]: 'Cosmic_noise'
}

for i, pair_info in enumerate(ccf_pairs_info):
    label1 = pair_info['label1']
    label2 = pair_info['label2']
    series1 = pair_info['series1_diff']
    series2 = pair_info['series2_diff']
    ax = axes_ccf_diff_flat[i]

    label1_disp = display_label_map.get(label1, label1)
    label2_disp = display_label_map.get(label2, label2)
    ax.set_title(f'Diff({label1_disp}) vs Diff({label2_disp})', fontsize=10)

    print(f"Processing Diff CCF for: {label1} vs {label2} (Length S1: {len(series1)}, S2: {len(series2)})")

    if len(series1) < max_lag_points_calc + 2 or len(series2) < max_lag_points_calc + 2 or \
            np.all(np.isnan(series1)) or np.all(np.isnan(series2)) or max_lag_points_calc == 0:
        msg = f"Skipping CCF for Diff({label1}) vs Diff({label2}): "
        if max_lag_points_calc == 0:
            msg += "max_lag_points is 0. "
        else:
            msg += "Insufficient or all-NaN data. "
        print(msg)
        ax.text(0.5, 0.5, "No Valid Data for CCF", ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_ylim(-0.2, 0.2);
        ax.grid(True, linestyle=':', linewidth=0.5)  # Fixed y-lim
        ax.set_xlim(-max_lag_minutes_ccf_plot - 5, max_lag_minutes_ccf_plot + 5)  # Ensure xlim is set
        continue

    lags, corrs, n_valid = calculate_ccf_robust(series1, series2, max_lag_points_calc)

    if lag_points_per_minute_final > 0 and not np.all(np.isnan(lags)):
        lags_minutes = lags / lag_points_per_minute_final
    else:
        lags_minutes = lags.astype(float)

    conf_interval = 1.96 / np.sqrt(n_valid) if n_valid > 0 else np.nan

    peak_corr_to_display = np.nan
    peak_lag_to_display = np.nan
    peak_text_prefix = "Peak"

    lags_minutes_np = np.asarray(lags_minutes)
    corrs_np = np.asarray(corrs)

    if not np.all(np.isnan(corrs_np)):
        if label1 == 'TSO_Mag' and label2 == 'Bz':
            bz_peak_search_mask = (lags_minutes_np >= 0) & \
                                  (lags_minutes_np <= 60) & \
                                  (corrs_np > 0) & \
                                  ~np.isnan(corrs_np)
            if np.any(bz_peak_search_mask):
                corrs_in_bz_range = corrs_np[bz_peak_search_mask]
                lags_in_bz_range = lags_minutes_np[bz_peak_search_mask]
                if len(corrs_in_bz_range) > 0:
                    peak_idx_local_bz = np.nanargmax(corrs_in_bz_range)
                    peak_corr_to_display = corrs_in_bz_range[peak_idx_local_bz]
                    peak_lag_to_display = lags_in_bz_range[peak_idx_local_bz]
                    peak_text_prefix = "Peak"
                    print(
                        f"  Bz Peak (Diff CCF, lag 0-60min, corr > 0): {peak_corr_to_display:.3f} at lag {peak_lag_to_display:.1f} min (N_overlap={n_valid})")
                else:
                    print(
                        f"  No valid positive correlations for Bz peak search in 0-60min range (Diff CCF, N_overlap={n_valid})")
            else:
                print(f"  No positive correlations found for Bz in lag 0-60min range (Diff CCF, N_overlap={n_valid})")
        else:
            search_lag_min_default = -50
            search_lag_max_default = 50
            search_lag_min_default = max(search_lag_min_default, -max_lag_minutes_ccf_plot)
            search_lag_max_default = min(search_lag_max_default, max_lag_minutes_ccf_plot)

            default_search_mask = (lags_minutes_np >= search_lag_min_default) & \
                                  (lags_minutes_np <= search_lag_max_default) & \
                                  ~np.isnan(corrs_np)
            if np.any(default_search_mask):
                corrs_in_default_range = corrs_np[default_search_mask]
                lags_in_default_range = lags_minutes_np[default_search_mask]
                if len(corrs_in_default_range) > 0:
                    peak_idx_local_default = np.nanargmax(np.abs(corrs_in_default_range))
                    peak_corr_to_display = corrs_in_default_range[peak_idx_local_default]
                    peak_lag_to_display = lags_in_default_range[peak_idx_local_default]
                    print(
                        f"  Peak ({search_lag_min_default} to {search_lag_max_default} min): {peak_corr_to_display:.3f} at lag {peak_lag_to_display:.1f} min (N_overlap={n_valid})")
                else:
                    print(f"  No valid data in default lag range for peak finding (N_overlap={n_valid})")
            else:
                print(
                    f"  No data in default lag range {search_lag_min_default} to {search_lag_max_default} min (N_overlap={n_valid})")
    else:
        print(f"  CCF calculation resulted in all NaNs for {label1} vs {label2}. Skipping peak finding.")

    if not np.all(np.isnan(corrs_np)):
        markerline, stemlines, baseline = ax.stem(lags_minutes_np, corrs_np, linefmt='b-', markerfmt='b.', basefmt='k-')
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=0.8)

        indices_to_highlight = np.zeros_like(lags_minutes_np, dtype=bool)
        if label1 == 'TSO_Mag' and label2 == riometer_labels_final[0]:
            indices_to_highlight = (corrs_np > 0) & ~np.isnan(corrs_np)
        elif label1 == 'TSO_Mag' and label2 == 'VHF_Ne':
            indices_to_highlight = (corrs_np < 0) & ~np.isnan(corrs_np)
        elif label1 == 'TSO_Mag' and label2 == 'UHF_Ne':
            indices_to_highlight = (corrs_np < 0) & ~np.isnan(corrs_np)
        elif label1 == 'TSO_Mag' and label2 == 'Bz':
            indices_to_highlight = (lags_minutes_np > 0) & (corrs_np > 0) & ~np.isnan(corrs_np)

        valid_highlight_indices = indices_to_highlight & ~np.isnan(lags_minutes_np)

        if np.any(valid_highlight_indices):
            lags_highlight = lags_minutes_np[valid_highlight_indices]
            corrs_highlight = corrs_np[valid_highlight_indices]
            markerline_h, stemlines_h, baseline_h = ax.stem(lags_highlight, corrs_highlight, linefmt='r-',
                                                            markerfmt='r.', basefmt='none')
            plt.setp(markerline_h, markersize=4, color='red')
            plt.setp(stemlines_h, color='red', linewidth=0.8)

        # --- MODIFIED Y-LIMIT SETTING for fig_ccf_diff ---
        ax.set_ylim(-0.2, 0.2)
        # --- END MODIFICATION ---

        if n_valid > 0 and not np.isnan(conf_interval):
            ax.axhline(conf_interval, color='grey', linestyle='--', linewidth=0.8, label='95% CI')
            ax.axhline(-conf_interval, color='grey', linestyle='--', linewidth=0.8)

        if not np.isnan(peak_corr_to_display) and not np.isnan(peak_lag_to_display):
            ax.text(0.95, 0.95, f'{peak_text_prefix}: {peak_corr_to_display:.2f}\nLag: {peak_lag_to_display:.1f}m',
                    va='top', ha='right', transform=ax.transAxes, color='black', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    else:
        # --- MODIFIED Y-LIMIT SETTING for fig_ccf_diff ---
        ax.set_ylim(-0.2, 0.2)
        # --- END MODIFICATION ---

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlim(-max_lag_minutes_ccf_plot - 5, max_lag_minutes_ccf_plot + 5)

fig_ccf_diff.text(0.5, 0.02, 'Lag (minutes)', ha='center', va='center')
fig_ccf_diff.text(0.04, 0.5, 'Correlation Coefficient', ha='center', va='center', rotation='vertical')
fig_ccf_diff.suptitle('Cross-Correlation Functions (Differenced Data)', fontsize=14)
plt.tight_layout(rect=[0.06, 0.04, 1, 0.94])
plt.savefig('Fig_2.png')
plt.show()
# plt.close(fig_ccf_diff)


# --- Combined Plot for RAW (Non-Differenced) CCF (TSO vs Others) ---
print("\n--- Calculating and Plotting Combined CCF (RAW/Processed Data - TSO vs Others) ---")
fig_ccf_raw, axes_ccf_raw = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes_ccf_raw_flat = axes_ccf_raw.flatten()

for i, pair_info in enumerate(ccf_pairs_info):
    label1 = pair_info['label1']
    label2 = pair_info['label2']
    series1 = pair_info['series1_raw']
    series2 = pair_info['series2_raw']
    ax = axes_ccf_raw_flat[i]

    label1_disp = display_label_map.get(label1, label1)
    label2_disp = display_label_map.get(label2, label2)
    ax.set_title(f'{label1_disp} vs {label2_disp}', fontsize=10)

    print(f"Processing Raw CCF for: {label1} vs {label2} (Length S1: {len(series1)}, S2: {len(series2)})")

    if len(series1) < max_lag_points_calc + 2 or len(series2) < max_lag_points_calc + 2 or \
            np.all(np.isnan(series1)) or np.all(np.isnan(series2)) or max_lag_points_calc == 0:
        msg = f"Skipping Raw CCF for {label1} vs {label2}: "
        if max_lag_points_calc == 0:
            msg += "max_lag_points is 0. "
        else:
            msg += "Insufficient or all-NaN data. "
        print(msg)
        ax.text(0.5, 0.5, "No Valid Data for CCF", ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_ylim(-0.5, 0.5);
        ax.grid(True, linestyle=':', linewidth=0.5)  # Original y-lim for raw
        ax.set_xlim(-max_lag_minutes_ccf_plot - 5, max_lag_minutes_ccf_plot + 5)
        continue

    lags, corrs, n_valid = calculate_ccf_robust(series1, series2, max_lag_points_calc)

    if lag_points_per_minute_final > 0 and not np.all(np.isnan(lags)):
        lags_minutes = lags / lag_points_per_minute_final
    else:
        lags_minutes = lags.astype(float)

    conf_interval = 1.96 / np.sqrt(n_valid) if n_valid > 0 else np.nan

    peak_corr_to_display_raw = np.nan
    peak_lag_to_display_raw = np.nan

    lags_minutes_np_raw = np.asarray(lags_minutes)
    corrs_np_raw = np.asarray(corrs)

    if not np.all(np.isnan(corrs_np_raw)):
        if label2 == 'Bz':
            search_lag_min_raw = max(-max_lag_minutes_ccf_plot, -60)
            search_lag_max_raw = min(max_lag_minutes_ccf_plot, 60)
        else:
            search_lag_min_raw = max(-max_lag_minutes_ccf_plot, -50)
            search_lag_max_raw = min(max_lag_minutes_ccf_plot, 50)

        raw_search_mask = (lags_minutes_np_raw >= search_lag_min_raw) & \
                          (lags_minutes_np_raw <= search_lag_max_raw) & \
                          ~np.isnan(corrs_np_raw)

        if np.any(raw_search_mask):
            corrs_in_raw_range = corrs_np_raw[raw_search_mask]
            lags_in_raw_range = lags_minutes_np_raw[raw_search_mask]
            if len(corrs_in_raw_range) > 0:
                peak_idx_local_raw = np.nanargmax(np.abs(corrs_in_raw_range))
                peak_corr_to_display_raw = corrs_in_raw_range[peak_idx_local_raw]
                peak_lag_to_display_raw = lags_in_raw_range[peak_idx_local_raw]
                print(
                    f"  Peak (Raw CCF, range {search_lag_min_raw} to {search_lag_max_raw} min): {peak_corr_to_display_raw:.3f} at lag {peak_lag_to_display_raw:.1f} min (N_overlap={n_valid})")
            else:
                print(f"  No valid data in specified lag range for raw peak finding (N_overlap={n_valid})")
        else:
            print(
                f"  No data points in specified lag range {search_lag_min_raw} to {search_lag_max_raw} min for raw peak finding (N_overlap={n_valid})")
    else:
        print(f"  Raw CCF calculation resulted in all NaNs for {label1} vs {label2}. Skipping peak finding.")

    valid_corrs_raw = corrs_np_raw[~np.isnan(corrs_np_raw)]
    min_corr_val_raw = np.min(valid_corrs_raw) if len(valid_corrs_raw) > 0 else -0.5
    max_corr_val_raw = np.max(valid_corrs_raw) if len(valid_corrs_raw) > 0 else 0.5

    if not np.all(np.isnan(corrs_np_raw)):
        markerline_raw, stemlines_raw, baseline_raw = ax.stem(lags_minutes_np_raw, corrs_np_raw, linefmt='b-',
                                                              markerfmt='b.', basefmt='k-')
        plt.setp(markerline_raw, markersize=4)
        plt.setp(stemlines_raw, linewidth=0.8)

        indices_to_highlight_raw = np.zeros_like(lags_minutes_np_raw, dtype=bool)
        if label1 == 'TSO_Mag' and label2 == riometer_labels_final[0]:
            indices_to_highlight_raw = (corrs_np_raw > 0) & ~np.isnan(corrs_np_raw)
        elif label1 == 'TSO_Mag' and label2 == 'VHF_Ne':
            indices_to_highlight_raw = (corrs_np_raw < 0) & ~np.isnan(corrs_np_raw)
        elif label1 == 'TSO_Mag' and label2 == 'UHF_Ne':
            indices_to_highlight_raw = (corrs_np_raw < 0) & ~np.isnan(corrs_np_raw)
        elif label1 == 'TSO_Mag' and label2 == 'Bz':
            indices_to_highlight_raw = (lags_minutes_np_raw > 0) & (corrs_np_raw > 0) & ~np.isnan(corrs_np_raw)

        valid_highlight_indices_raw = indices_to_highlight_raw & ~np.isnan(lags_minutes_np_raw)

        if np.any(valid_highlight_indices_raw):
            lags_highlight_raw = lags_minutes_np_raw[valid_highlight_indices_raw]
            corrs_highlight_raw = corrs_np_raw[valid_highlight_indices_raw]
            markerline_h_raw, stemlines_h_raw, baseline_h_raw = ax.stem(lags_highlight_raw, corrs_highlight_raw,
                                                                        linefmt='r-',
                                                                        markerfmt='r.', basefmt='none')
            plt.setp(markerline_h_raw, markersize=4, color='red')
            plt.setp(stemlines_h_raw, color='red', linewidth=0.8)

        # Original y-limit calculation for Raw CCF plots
        if n_valid > 0 and not np.isnan(conf_interval):
            ax.axhline(conf_interval, color='grey', linestyle='--', linewidth=0.8, label='95% CI')
            ax.axhline(-conf_interval, color='grey', linestyle='--', linewidth=0.8)
            y_abs_max_raw = max(np.abs(min_corr_val_raw), np.abs(max_corr_val_raw),
                                conf_interval if not np.isnan(conf_interval) else 0)
            y_lim_val_raw = y_abs_max_raw * 1.1 if y_abs_max_raw > 0 else 0.5
        else:
            y_abs_max_raw = max(np.abs(min_corr_val_raw), np.abs(max_corr_val_raw))
            y_lim_val_raw = y_abs_max_raw * 1.1 if y_abs_max_raw > 0 else 0.5

        y_min_plot_raw = -y_lim_val_raw
        y_max_plot_raw = y_lim_val_raw
        if np.isnan(y_min_plot_raw) or np.isnan(y_max_plot_raw) or y_min_plot_raw >= y_max_plot_raw:
            y_min_plot_raw, y_max_plot_raw = -0.5, 0.5
        ax.set_ylim(y_min_plot_raw, y_max_plot_raw)  # Apply dynamic y-limits for raw plots

        if not np.isnan(peak_corr_to_display_raw) and not np.isnan(peak_lag_to_display_raw):
            ax.text(0.95, 0.95, f'Peak: {peak_corr_to_display_raw:.2f}\nLag: {peak_lag_to_display_raw:.1f}m',
                    va='top', ha='right', transform=ax.transAxes, color='black', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    else:
        ax.set_ylim(-0.5, 0.5)

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_xlim(-max_lag_minutes_ccf_plot - 5, max_lag_minutes_ccf_plot + 5)

fig_ccf_raw.text(0.5, 0.02, 'Lag (minutes)', ha='center', va='center')
fig_ccf_raw.text(0.04, 0.5, 'Correlation Coefficient', ha='center', va='center', rotation='vertical')
fig_ccf_raw.suptitle('Cross-Correlation Functions (Raw/Processed Data)', fontsize=14)
plt.tight_layout(rect=[0.06, 0.04, 1, 0.94])

plt.show()
# plt.close(fig_ccf_raw)


# --- Combined Plot for AUTOCORRELATION (Raw/Processed Data - Reduced) ---
print("\n--- Calculating and Plotting Combined ACF (RAW/Processed Data - Reduced) ---")
acf_datasets_info = [
    {'label': display_label_map.get(riometer_labels_final[0], riometer_labels_final[0]), 'series': skn_30_ccf},
    {'label': display_label_map.get('TSO_Mag', 'TSO_Mag'), 'series': tso_mag_ccf},
    {'label': display_label_map.get('VHF_Ne', 'VHF_Ne'), 'series': vhf_ne_ccf},
    {'label': display_label_map.get('UHF_Ne', 'UHF_Ne'), 'series': uhf_ne_ccf},
    {'label': display_label_map.get('Bz', 'Bz'), 'series': bz_ccf}
]

lags_acf_hours = 3
if time_step_hours_final > 0:
    lags_acf = int(round(lags_acf_hours / time_step_hours_final))
    max_allowable_acf_lags = num_points_final - 1
    if lags_acf >= max_allowable_acf_lags:
        print(
            f"Warning: Requested ACF lags ({lags_acf_hours} hrs -> {lags_acf} pts) equals or exceeds series length ({num_points_final}). Reducing lags to {max_allowable_acf_lags - 1}.")
        lags_acf = max(1, max_allowable_acf_lags - 1)
        lags_acf_hours = lags_acf * time_step_hours_final
        print(f"Adjusted max ACF lags: {lags_acf} points (approx {lags_acf_hours:.1f} hours)")
    elif lags_acf < 1:
        print(f"Warning: Calculated ACF lags ({lags_acf}) is less than 1. Setting to 1.")
        lags_acf = 1
        lags_acf_hours = lags_acf * time_step_hours_final if time_step_hours_final > 0 else 0
else:
    print("Warning: Final time step is zero. Cannot calculate ACF lags reliably. Setting lags_acf=50.")
    lags_acf = 50

print(f"Calculating ACF up to {lags_acf} lags (approx. {lags_acf_hours:.1f} hours)")

fig_acf, axes_acf = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
axes_acf_flat = axes_acf.flatten()

for i, data_info in enumerate(acf_datasets_info):
    label = data_info['label']
    series = data_info['series']
    ax = axes_acf_flat[i]
    print(f"Processing ACF for: {label}")

    series_np = np.asarray(series)

    if len(series_np) < 2 or np.all(np.isnan(series_np)):
        print("  Skipping ACF: Series has < 2 points or contains only NaNs.")
        ax.set_title(f'ACF: {label} (No Data)', fontsize=10)
        ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_ylim(-0.2, 1.05);
        ax.grid(True, linestyle=':', linewidth=0.5)
        continue

    series_pd = pd.Series(series_np)
    nan_frac = series_pd.isna().mean()
    missing_opt = 'none'
    series_to_plot_acf = series_pd

    if nan_frac > 0:
        print(f"  Warning: Series contains {nan_frac * 100:.1f}% NaNs. Using missing='drop' for ACF.")
        missing_opt = 'drop'
        series_to_plot_acf = series_pd.dropna()

    if len(series_to_plot_acf) <= 1:
        print("  Skipping ACF: Series has <= 1 point after NaN handling.")
        ax.set_title(f'ACF: {label} (Too Few Points)', fontsize=10)
        ax.text(0.5, 0.5, "<=1 non-NaN pts", ha='center', va='center', transform=ax.transAxes, color='orange')
        ax.set_ylim(-0.2, 1.05);
        ax.grid(True, linestyle=':', linewidth=0.5)
        continue

    if np.std(series_to_plot_acf) < 1e-10:
        print("  Skipping ACF: Series is constant after NaN handling.")
        ax.set_title(f'ACF: {label} (Constant)', fontsize=10)
        ax.text(0.5, 0.5, "Series is Constant", ha='center', va='center', transform=ax.transAxes, color='orange')
        ax.set_ylim(-0.2, 1.05);
        ax.grid(True, linestyle=':', linewidth=0.5)
        continue

    current_lags_acf = min(lags_acf, len(series_to_plot_acf) - 1)
    if current_lags_acf < 1:
        print(
            f"  Skipping ACF for {label}: Not enough non-NaN points ({len(series_to_plot_acf)}) for any lags ({current_lags_acf}).")
        ax.set_title(f'ACF: {label} (Few Pts for Lags)', fontsize=10)
        ax.text(0.5, 0.5, f"<{current_lags_acf + 1} non-NaN", ha='center', va='center', transform=ax.transAxes,
                color='orange')
        ax.set_ylim(-0.2, 1.05);
        ax.grid(True, linestyle=':', linewidth=0.5)
        continue

    try:
        plot_acf(series_pd, ax=ax, lags=current_lags_acf, title=f'ACF: {label}', missing=missing_opt, fft=True,
                 zero=True)
        ax.set_ylim(-0.2, 1.05)
        ax.grid(True, linestyle=':', linewidth=0.5)
    except Exception as e:
        print(f"  Error plotting ACF for {label}: {e}")
        ax.set_title(f'ACF: {label} (Error)', fontsize=10)
        ax.text(0.5, 0.5, f"Plotting Error", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8)
        ax.set_ylim(-0.2, 1.05);
        ax.grid(True, linestyle=':', linewidth=0.5)

if len(acf_datasets_info) < len(axes_acf_flat):
    for i in range(len(acf_datasets_info), len(axes_acf_flat)):
        axes_acf_flat[i].set_visible(False)

fig_acf.text(0.5, 0.02, f'Lag (Points, 1 lag ≈ {time_step_hours_final * 60:.1f} min)', ha='center', va='center')
fig_acf.text(0.04, 0.5, 'Autocorrelation', ha='center', va='center', rotation='vertical')
fig_acf.suptitle('Autocorrelation Functions (Raw/Processed Data)', fontsize=14)
plt.tight_layout(rect=[0.06, 0.04, 1, 0.94])
# plt.savefig('plots/acf_combined_raw_reduced.png')
plt.show()
# plt.close(fig_acf)


# --- NEW: SEPARATE 2-PANEL PLOT (SKN_30, TSO_Mag) ---
print("\n--- Generating separate 2-panel plot (SKN_30, TSO_Mag) ---")


def format_coord_main_builder(ax_series, ax_label, time_axis_ref):
    series_np = np.asarray(ax_series)
    time_axis_np = np.asarray(time_axis_ref)

    def format_coord(x, y):
        ut_hour = (x + 9) % 24
        if len(time_axis_np) > 0:
            if x < time_axis_np[0] or x > time_axis_np[-1]:
                val_str = "N/A (out of time bounds)"
            else:
                idx = np.argmin(np.abs(time_axis_np - x));
                if idx < len(series_np):
                    actual_value = series_np[idx]
                    if np.isnan(actual_value):
                        val_str = "NaN"
                    elif abs(actual_value) > 1e4 or (0 < abs(actual_value) < 1e-2 and actual_value != 0):
                        val_str = f'{actual_value:.2e}'
                    else:
                        val_str = f'{actual_value:.2f}'
                else:
                    val_str = "N/A (index)"
        else:
            val_str = "N/A (no time)"
        return f'Time: {ut_hour:.2f} UT ({x:.2f} hrs), {ax_label}: {val_str}'

    return format_coord


if len(hours_final) > 0:
    fig_skn_tso, axes_skn_tso = plt.subplots(nrows=2, ncols=1, figsize=(12, 2.5 * 2), sharex=True)

    skn_data = riometer_data_final[0]
    skn_label_display = "Skibotn riometer 30 MHz"
    axes_skn_tso[0].plot(hours_final, skn_data, 'b-', label=skn_label_display)
    axes_skn_tso[0].set_ylabel('Absorption (dB)')
    axes_skn_tso[0].grid(True)
    axes_skn_tso[0].legend(loc='upper right')
    axes_skn_tso[0].set_title(f'{skn_label_display} and Tromsø Magnetometer H-component')
    axes_skn_tso[0].format_coord = format_coord_main_builder(skn_data, skn_label_display, hours_final)

    tso_label_display = "Tromsø H-component"
    axes_skn_tso[1].plot(hours_final, tso_mag_plot, 'r-', label=tso_label_display)
    axes_skn_tso[1].set_ylabel('H (nT)')
    axes_skn_tso[1].grid(True)
    axes_skn_tso[1].legend(loc='upper right')
    axes_skn_tso[1].format_coord = format_coord_main_builder(tso_mag_plot, tso_label_display, hours_final)

    axes_skn_tso[1].set_xlabel('Time (UT)')
    axes_skn_tso[1].set_xticks(np.arange(0, 25, 1))
    if 'tick_labels' not in locals() or tick_labels is None:
        tick_labels = [str((h + 9) % 24) for h in range(25)]
    axes_skn_tso[1].set_xticklabels(tick_labels)
    axes_skn_tso[1].set_xlim(0, 24)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # plt.savefig('skn_tso_comparison_plot.png')
    plt.show()
    # plt.close(fig_skn_tso)

else:
    print("Skipping SKN_30 / TSO Mag plot: No time axis data (hours_final is empty).")

# --- FINAL 5-PANEL PLOT (SKN, TSO, EISCAT, ACE) ---
print("\n--- Generating final 5-panel plot with VISUAL alignments (SKN, TSO, EISCAT, ACE) ---")

if len(hours_final) > 0:
    fig_main, axes_main = plt.subplots(nrows=5, ncols=1, figsize=(12, 2.5 * 5), sharex=True)

    skn_label_main_plot = "Skibotn riometer 30MHz"
    axes_main[0].plot(hours_final, riometer_data_final[0], 'b-', label=skn_label_main_plot)
    axes_main[0].set_ylabel('Cosmic noise (arb. units)')
    axes_main[0].grid(True)
    axes_main[0].legend(loc='upper right')
    axes_main[0].set_title('Cosmic noise, H-component, Electron density, and IMF Bz')

    tso_label_main_plot = "Tromsø H-component"
    axes_main[1].plot(hours_final, tso_mag_plot, 'r-', label=tso_label_main_plot)
    axes_main[1].set_ylabel('H (nT)')
    axes_main[1].grid(True)
    axes_main[1].legend(loc='upper right')

    vhf_label_main_plot = "Ne (E-region, EISCAT-VHF)"
    axes_main[2].plot(hours_final, vhf_ne_plot, 'b-', label=vhf_label_main_plot)
    axes_main[2].set_ylabel('Ne (m^-3)')
    axes_main[2].set_yscale('log')
    axes_main[2].grid(True)
    axes_main[2].legend(loc='upper right')

    uhf_label_main_plot = "Ne (E-region, EISCAT-UHF)"
    axes_main[3].plot(hours_final, uhf_ne_plot, 'b-', label=uhf_label_main_plot)
    axes_main[3].set_ylabel('Ne (m^-3)')
    axes_main[3].set_yscale('log')
    axes_main[3].grid(True)
    axes_main[3].legend(loc='upper right')

    bz_plot_np = np.asarray(bz_plot)
    if len(bz_plot_np[np.isfinite(bz_plot_np)]) > 1:
        points = np.array([hours_final, bz_plot_np]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        colors_bz = []
        valid_segments_bz = []
        for i in range(len(bz_plot_np) - 1):
            bz1 = bz_plot_np[i]
            bz2 = bz_plot_np[i + 1]
            if np.isfinite(bz1) and np.isfinite(bz2):
                valid_segments_bz.append(segments[i])
                colors_bz.append('g' if (bz1 + bz2) / 2.0 >= 0 else 'r')

        if valid_segments_bz:
            lc = LineCollection(valid_segments_bz, colors=colors_bz, linewidth=2)
            axes_main[4].add_collection(lc)
            axes_main[4].autoscale_view()
        else:
            print("No valid segments to plot for ACE Bz LineCollection.")
            axes_main[4].set_ylim(-10, 10)
    else:
        print("Insufficient valid data points for ACE Bz LineCollection.")
        axes_main[4].set_ylim(-10, 10)

    axes_main[4].plot([], [], 'g-', label='Bz > 0')
    axes_main[4].plot([], [], 'r-', label='Bz < 0')
    axes_main[4].set_ylabel('Bz (nT)')
    axes_main[4].grid(True)
    axes_main[4].legend(loc='upper right')

    axes_main[4].set_xlabel('Time (UT)')
    axes_main[4].set_xticks(np.arange(0, 25, 1))
    if 'tick_labels' not in locals() or tick_labels is None:
        tick_labels = [str((h + 9) % 24) for h in range(25)]
    axes_main[4].set_xticklabels(tick_labels)
    axes_main[4].set_xlim(0, 24)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    axes_main[0].format_coord = format_coord_main_builder(riometer_data_final[0], skn_label_main_plot, hours_final)
    axes_main[1].format_coord = format_coord_main_builder(tso_mag_plot, tso_label_main_plot, hours_final)
    axes_main[2].format_coord = format_coord_main_builder(vhf_ne_plot, vhf_label_main_plot, hours_final)
    axes_main[3].format_coord = format_coord_main_builder(uhf_ne_plot, uhf_label_main_plot, hours_final)
    axes_main[4].format_coord = format_coord_main_builder(bz_plot, "ACE Bz", hours_final)

    # plt.savefig('Fig_3.png')
    plt.show()
    # plt.close(fig_main)

else:
    print("Skipping main 5-panel plot: No time axis data (hours_final is empty).")

print("\n--- Script Finished ---")