import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection
import scipy.io

# Define the time range
start_time = datetime(2025, 3, 3, 9, 0, 0)  # 9 UT March 3, 2025
end_time = datetime(2025, 3, 4, 9, 0, 0)    # 9 UT March 4, 2025

# Load ACE magnetometer data (Bz component) for both days
ace_files = ['20250303_ace_mag_1m.txt', '20250304_ace_mag_1m.txt']
ace_datetimes = []
ace_bz = []
ace_status = []

for ace_file in ace_files:
    data = np.loadtxt(ace_file, skiprows=14, usecols=(0, 1, 2, 3, 5, 6, 9), dtype=str)
    for row in data:
        dt = datetime.strptime(f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]}", "%Y %m %d %H%M")
        if start_time <= dt <= end_time:
            ace_datetimes.append(dt)
            ace_status.append(int(row[5]))
            ace_bz.append(float(row[6]))

ace_datetimes = np.array(ace_datetimes)
ace_status = np.array(ace_status)
ace_bz = np.array(ace_bz)

# Filter valid ACE data (status = 0 and Bz != -999.9)
valid_ace = (ace_status == 0) & (ace_bz != -999.9)
ace_datetimes = ace_datetimes[valid_ace]
ace_bz = ace_bz[valid_ace]
Bz_delay = 2.8
ace_hours = np.array([(dt - start_time).total_seconds() / 3600.0 + Bz_delay for dt in ace_datetimes])

# Load magnetometer data (horizontal component) for NAL
nal_files = ['NAL.txt', 'NAL_2.txt']
nal_datetimes = []
nal_mag_horizontal = []

for nal_file in nal_files:
    data = np.loadtxt(nal_file, skiprows=6, usecols=(0, 1, 3), dtype=str)
    for row in data:
        dt = datetime.strptime(f"{row[0]} {row[1]}", "%d/%m/%Y %H:%M:%S")
        if start_time <= dt <= end_time:
            nal_datetimes.append(dt)
            nal_mag_horizontal.append(float(row[2]))

nal_datetimes = np.array(nal_datetimes)
nal_mag_horizontal = np.array(nal_mag_horizontal)
nal_mag_hours = np.array([(dt - start_time).total_seconds() / 3600.0 for dt in nal_datetimes])
nal_mag_delay = 1.39
nal_mag_hours = nal_mag_hours + nal_mag_delay

# Load magnetometer data (horizontal component) for TSO
tso_files = ['TSO.txt', 'TSO_2.txt']
tso_datetimes = []
tso_mag_horizontal = []

for tso_file in tso_files:
    data = np.loadtxt(tso_file, skiprows=6, usecols=(0, 1, 3), dtype=str)
    for row in data:
        dt = datetime.strptime(f"{row[0]} {row[1]}", "%d/%m/%Y %H:%M:%S")
        if start_time <= dt <= end_time:
            tso_datetimes.append(dt)
            tso_mag_horizontal.append(float(row[2]))

tso_datetimes = np.array(tso_datetimes)
tso_mag_horizontal = np.array(tso_mag_horizontal)
tso_mag_hours = np.array([(dt - start_time).total_seconds() / 3600.0 for dt in tso_datetimes])
tso_mag_delay = 0.40
tso_mag_hours = tso_mag_hours + tso_mag_delay

# Load EISCAT VHF electron density data
file_path_vhf = "bella-20250303-20250304.mat"
mat_data_vhf = scipy.io.loadmat(file_path_vhf)
ne_vhf = mat_data_vhf.get('ne')
h_vhf = mat_data_vhf.get('h')
T_vhf = mat_data_vhf.get('T')

# Load EISCAT UHF electron density data
file_path_uhf = "beata-20250303-20250304.mat"
mat_data_uhf = scipy.io.loadmat(file_path_uhf)
ne_uhf = mat_data_uhf.get('ne')
h_uhf = mat_data_uhf.get('h')
T_uhf = mat_data_uhf.get('T')

# Convert EISCAT VHF time array to datetime and filter
eiscat_times_vhf = []
eiscat_indices_vhf = []
for i, t in enumerate(T_vhf):
    year, month, day, hour, minute, second = t
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    if start_time <= dt <= end_time:
        eiscat_times_vhf.append(dt)
        eiscat_indices_vhf.append(i)
eiscat_times_vhf = np.array(eiscat_times_vhf)
eiscat_indices_vhf = np.array(eiscat_indices_vhf)

# Convert EISCAT UHF time array to datetime and filter
eiscat_times_uhf = []
eiscat_indices_uhf = []
for i, t in enumerate(T_uhf):
    year, month, day, hour, minute, second = t
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    if start_time <= dt <= end_time:
        eiscat_times_uhf.append(dt)
        eiscat_indices_uhf.append(i)
eiscat_times_uhf = np.array(eiscat_times_uhf)
eiscat_indices_uhf = np.array(eiscat_indices_uhf)

# Convert EISCAT times to hours (relative to start_time)
eiscat_hours_vhf = np.array([(t - start_time).total_seconds() / 3600.0 for t in eiscat_times_vhf])
eiscat_hours_uhf = np.array([(t - start_time).total_seconds() / 3600.0 for t in eiscat_times_uhf])

# Prepare EISCAT VHF data
h_vhf = h_vhf.flatten()
mask_vhf = (h_vhf >= 90) & (h_vhf <= 150)
h_filtered_vhf = h_vhf[mask_vhf]
ne_filtered_vhf = ne_vhf[mask_vhf, :][:, eiscat_indices_vhf]
ne_filtered_vhf = np.where(ne_filtered_vhf > 0, ne_filtered_vhf, np.nan)
ne_filtered_vhf = np.where(np.isfinite(ne_filtered_vhf), ne_filtered_vhf, np.nan)
median_ne_vhf = np.nanmedian(ne_filtered_vhf, axis=0)

# Prepare EISCAT UHF data
h_uhf = h_uhf.flatten()
mask_uhf = (h_uhf >= 90) & (h_uhf <= 150)
h_filtered_uhf = h_uhf[mask_uhf]
ne_filtered_uhf = ne_uhf[mask_uhf, :][:, eiscat_indices_uhf]
ne_filtered_uhf = np.where(ne_filtered_uhf > 0, ne_filtered_uhf, np.nan)
ne_filtered_uhf = np.where(np.isfinite(ne_filtered_uhf), ne_filtered_uhf, np.nan)
median_ne_uhf = np.nanmedian(ne_filtered_uhf, axis=0)

# Compute rolling median for VHF
window_size = 15
rolling_median_vhf = np.full_like(median_ne_vhf, np.nan)
for i in range(window_size - 1, len(median_ne_vhf)):
    window = median_ne_vhf[i - window_size + 1:i + 1]
    rolling_median_vhf[i] = np.nanmedian(window)

# Compute rolling median for UHF
rolling_median_uhf = np.full_like(median_ne_uhf, np.nan)
for i in range(window_size - 1, len(median_ne_uhf)):
    window = median_ne_uhf[i - window_size + 1:i + 1]
    rolling_median_uhf[i] = np.nanmedian(window)

# List of riometer instruments
riometer_instruments = [
    ('NAL_30', ['NAL_30.txt', 'NAL_30_2.txt']),
    ('NAL_40', ['NAL_40.txt', 'NAL_40_2.txt']),
    ('SKN_30', ['SKN_30.txt', 'SKN_30_2.txt'])
]

# Function to convert Unix timestamps to hours
def timestamps_to_hours(timestamps, t0):
    return [(t - t0) / 3600.0 for t in timestamps]

# Function to compute median every n samples
def median_every_n(data, n):
    n = int(n)
    trimmed_data = data[:len(data) - (len(data) % n)]
    reshaped_data = trimmed_data.reshape(-1, n)
    return np.median(reshaped_data, axis=1)

# Prepare riometer data
hours_common = np.linspace(0, 24, 41935)
values_list = []
labels = []
n_median = 60

for instrument, files in riometer_instruments:
    timestamps = []
    values = []
    for file in files:
        data = np.loadtxt(file, usecols=(0, 1))
        file_timestamps = data[:, 0]
        file_values = data[:, 1]
        file_datetimes = [datetime.utcfromtimestamp(ts) for ts in file_timestamps]
        for dt, val in zip(file_datetimes, file_values):
            if start_time <= dt <= end_time:
                timestamps.append((dt - start_time).total_seconds() / 3600.0)
                values.append(val)
    timestamps = np.array(timestamps)
    values = np.array(values)
    sorted_indices = np.argsort(timestamps)
    timestamps = timestamps[sorted_indices]
    values = values[sorted_indices]
    interp_func = interp1d(timestamps, values, kind='linear', fill_value='extrapolate')
    values_interp = interp_func(hours_common)
    values_median = median_every_n(values_interp, n_median)
    values_list.append(values_median)
    labels.append(instrument)

# Create new time grid for median data
num_points_median = len(values_list[0])
hours_median = np.linspace(0, 24, num_points_median)

# Interpolate magnetometer, EISCAT, and ACE Bz data
nal_mag_interp_func = interp1d(nal_mag_hours, nal_mag_horizontal, kind='linear', fill_value='extrapolate')
nal_mag_interp = nal_mag_interp_func(hours_median)
nal_mag_interp[hours_median < nal_mag_delay] = np.nan

tso_mag_interp_func = interp1d(tso_mag_hours, tso_mag_horizontal, kind='linear', fill_value='extrapolate')
tso_mag_interp = tso_mag_interp_func(hours_median)
tso_mag_interp[hours_median < tso_mag_delay] = np.nan

eiscat_vhf_interp_func = interp1d(eiscat_hours_vhf, rolling_median_vhf, kind='linear', fill_value='extrapolate')
eiscat_vhf_interp = eiscat_vhf_interp_func(hours_median)
eiscat_vhf_interp[hours_median < 0] = np.nan

eiscat_uhf_interp_func = interp1d(eiscat_hours_uhf, rolling_median_uhf, kind='linear', fill_value='extrapolate')
eiscat_uhf_interp = eiscat_uhf_interp_func(hours_median)
eiscat_uhf_interp[hours_median < 0] = np.nan

ace_bz_interp_func = interp1d(ace_hours, ace_bz, kind='linear', fill_value='extrapolate')
ace_bz_interp = ace_bz_interp_func(hours_median)
ace_bz_interp[hours_median < Bz_delay] = np.nan

# Create stacked subplots
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(12, 2.5 * 8), sharex=True)

# Plot NAL_30 riometer
axes[0].plot(hours_median, values_list[0], 'b-', label='NAL_30')
axes[0].set_ylabel('Value')
axes[0].grid(True)
axes[0].legend(loc='upper right')
axes[0].set_title('Riometer, Magnetometer, Electron Density, and Solar Wind Data (March 3, 2025 9 UT to March 4, 2025 9 UT)')

# Plot NAL_40 riometer
axes[1].plot(hours_median, values_list[1], 'b-', label='NAL_40')
axes[1].set_ylabel('Value')
axes[1].grid(True)
axes[1].legend(loc='upper right')

# Plot NAL magnetometer
axes[2].plot(hours_median, nal_mag_interp, 'r-', label='NAL Magnetometer (Horizontal)')
axes[2].set_ylabel('H (nT)')
axes[2].grid(True)
axes[2].legend(loc='upper right')

# Plot SKN_30 riometer
axes[3].plot(hours_median, values_list[2], 'b-', label='SKN_30')
axes[3].set_ylabel('Value')
axes[3].grid(True)
axes[3].legend(loc='upper right')

# Plot TSO magnetometer
axes[4].plot(hours_median, tso_mag_interp, 'r-', label='TSO Magnetometer (Horizontal)')
axes[4].set_ylabel('H (nT)')
axes[4].grid(True)
axes[4].legend(loc='upper right')

# Plot EISCAT VHF rolling median electron density
axes[5].plot(hours_median, eiscat_vhf_interp, 'b-', label='Ne (E-region, EISCAT-VHF)')
axes[5].set_ylabel('Ne (m^-3)')
axes[5].set_yscale('log')
axes[5].grid(True)
axes[5].legend(loc='upper right')

# Plot EISCAT UHF rolling median electron density
axes[6].plot(hours_median, eiscat_uhf_interp, 'b-', label='Ne (E-region, EISCAT-UHF)')
axes[6].set_ylabel('Ne (m^-3)')
axes[6].set_yscale('log')
axes[6].grid(True)
axes[6].legend(loc='upper right')

# Plot ACE Bz with color coding
points = np.array([hours_median, ace_bz_interp]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
colors = ['g' if (ace_bz_interp[i] + ace_bz_interp[i + 1]) / 2 >= 0 else 'r' for i in range(len(ace_bz_interp) - 1)]
lc = LineCollection(segments, colors=colors, linewidth=2, label='ACE Bz (GSM)')
axes[7].add_collection(lc)
axes[7].plot([], [], 'g-', label='ACE Bz (GSM) > 0')
axes[7].plot([], [], 'r-', label='ACE Bz (GSM) < 0')
axes[7].set_ylabel('Bz (nT)')
axes[7].grid(True)
axes[7].legend(loc='upper right')
axes[7].autoscale()

# Set x-axis properties
axes[7].set_xlabel('Time (UT)')
axes[7].set_xticks(np.arange(0, 25, 1))
tick_labels = [str((h + 9) % 24) for h in range(25)]
axes[7].set_xticklabels(tick_labels)
axes[7].set_xlim(0, 24)

# Adjust layout
plt.tight_layout()
# Before plt.show()
def format_coord(x, y):
    return f'Time: {(x+9):.2f} hours, Value: {y:.2f}'

axes[0].format_coord = format_coord  # Apply to the bottom subplot
axes[1].format_coord = format_coord  # Apply to the bottom subplot
axes[2].format_coord = format_coord  # Apply to the bottom subplot
axes[3].format_coord = format_coord  # Apply to the bottom subplot
axes[4].format_coord = format_coord  # Apply to the bottom subplot
axes[5].format_coord = format_coord  # Apply to the bottom subplot
axes[6].format_coord = format_coord  # Apply to the bottom subplot
axes[7].format_coord = format_coord  # Apply to the bottom subplot
# Save and close
plt.show()
#plt.savefig('plots/all_measurements_9ut_march3_to_9ut_march4.png')
#plt.close()