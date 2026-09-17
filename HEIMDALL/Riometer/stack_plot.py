import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
from matplotlib.collections import LineCollection
import scipy.io

# Load ACE magnetometer data (Bz component)
ace_data = np.loadtxt('20250303_ace_mag_1m.txt', skiprows=14, usecols=(0, 1, 2, 3, 5, 6, 9), dtype=str)
ace_datetimes = [datetime.strptime(f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]}", "%Y %m %d %H%M") for row in ace_data]
ace_status = np.array([int(row[5]) for row in ace_data])
ace_bz = np.array([float(row[6]) for row in ace_data])

# Filter valid ACE data (status = 0 and Bz != -999.9)
valid_ace = (ace_status == 0) & (ace_bz != -999.9)
ace_datetimes = np.array(ace_datetimes)[valid_ace]
ace_bz = ace_bz[valid_ace]
Bz_delay = 2.8
ace_hours = np.array([(dt.hour + dt.minute / 60 + dt.second / 3600 + Bz_delay) for dt in ace_datetimes])

# Load magnetometer data (horizontal component) for NAL
nal_mag_data = np.loadtxt('NAL.txt', skiprows=6, usecols=(0, 1, 3), dtype=str)
nal_datetimes = [datetime.strptime(f"{row[0]} {row[1]}", "%d/%m/%Y %H:%M:%S") for row in nal_mag_data]
nal_mag_horizontal = np.array([float(row[2]) for row in nal_mag_data])
nal_mag_hours = np.array([(dt.hour + dt.minute / 60 + dt.second / 3600) for dt in nal_datetimes])
nal_mag_hours = nal_mag_hours + 1.39  # Delay NAL magnetometer by 1.39 hours

# Load magnetometer data (horizontal component) for TSO
tso_mag_data = np.loadtxt('TSO.txt', skiprows=6, usecols=(0, 1, 3), dtype=str)
tso_datetimes = [datetime.strptime(f"{row[0]} {row[1]}", "%d/%m/%Y %H:%M:%S") for row in tso_mag_data]
tso_mag_horizontal = np.array([float(row[2]) for row in tso_mag_data])
tso_mag_hours = np.array([(dt.hour + dt.minute / 60 + dt.second / 3600) for dt in tso_datetimes])
tso_mag_hours = tso_mag_hours + 0.40  # Delay TSO magnetometer by 0.40 hours

# Load EISCAT electron density data
file_path = "bella-20250303-20250304.mat"
mat_data = scipy.io.loadmat(file_path)
ne = mat_data.get('ne')  # electron density (39, 2869)
h = mat_data.get('h')    # altitude (39, 1)
T = mat_data.get('T')    # time (2869, 6)

# Convert EISCAT time array to datetime
eiscat_times = []
for t in T:
    year, month, day, hour, minute, second = t
    eiscat_times.append(datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)))
eiscat_times = np.array(eiscat_times)

# Convert EISCAT times to hours (relative to start of March 3, 2025)
eiscat_hours = np.array([(t - eiscat_times[0]).total_seconds() / 3600.0 for t in eiscat_times])

# Prepare EISCAT data
h = h.flatten()  # Convert altitude to 1D array
mask = (h >= 90) & (h <= 150)
h_filtered = h[mask]
ne_filtered = ne[mask, :]  # Select corresponding rows of electron density

# Clean electron density data
ne_filtered = np.where(ne_filtered > 0, ne_filtered, np.nan)  # Replace non-positive values with NaN
ne_filtered = np.where(np.isfinite(ne_filtered), ne_filtered, np.nan)  # Replace inf/-inf with NaN

# Compute median electron density for each time step
median_ne = np.nanmedian(ne_filtered, axis=0)  # Median along altitude dimension

# Compute rolling median over 10 time samples
window_size = 15
rolling_median = np.full_like(median_ne, np.nan)
for i in range(window_size - 1, len(median_ne)):
    window = median_ne[i - window_size + 1:i + 1]
    rolling_median[i] = np.nanmedian(window)

# List of riometer instruments to plot
riometer_instruments = ['NAL_30', 'NAL_40', 'SKN_30']

# Function to convert Unix timestamps to relative hours
def timestamps_to_hours(timestamps):
    t0 = timestamps[0]
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

for instrument in riometer_instruments:
    data = np.loadtxt(f'{instrument}.txt', usecols=(0, 1))
    timestamps = data[:, 0]
    values = data[:, 1]
    hours = timestamps_to_hours(timestamps)
    interp_func = interp1d(hours, values, kind='linear', fill_value='extrapolate')
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
nal_mag_interp[hours_median < 1.39] = np.nan  # Mask before 1.39 hours

tso_mag_interp_func = interp1d(tso_mag_hours, tso_mag_horizontal, kind='linear', fill_value='extrapolate')
tso_mag_interp = tso_mag_interp_func(hours_median)
tso_mag_interp[hours_median < 0.40] = np.nan  # Mask before 0.40 hours

# Interpolate EISCAT rolling median electron density
eiscat_interp_func = interp1d(eiscat_hours, rolling_median, kind='linear', fill_value='extrapolate')
eiscat_interp = eiscat_interp_func(hours_median)
eiscat_interp[hours_median < 0] = np.nan  # Mask if needed (optional, adjust based on data availability)

ace_bz_interp_func = interp1d(ace_hours, ace_bz, kind='linear', fill_value='extrapolate')
ace_bz_interp = ace_bz_interp_func(hours_median)
ace_bz_interp[hours_median < Bz_delay] = np.nan  # Mask before 2.8 hours

# Create stacked subplots (7 subplots: NAL_30, NAL_40, NAL mag, SKN_30, TSO mag, EISCAT Ne, ACE Bz)
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 2.5 * 7), sharex=True)

# Plot NAL_30 riometer
axes[0].plot(hours_median, values_list[0], 'b-', label='NAL_30')
axes[0].set_ylabel('Value')
axes[0].grid(True)
axes[0].legend(loc='upper right')
axes[0].set_title('Riometer, Magnetometer, Electron Density, and Solar Wind Data (March 3, 2025 0 UT to March 4, 2025 0 UT)')

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

# Plot EISCAT rolling median electron density
axes[5].plot(hours_median, eiscat_interp, 'b-', label='EISCAT Median Ne (90-150 km, 10-sample window)')
axes[5].set_ylabel('Ne (m^-3)')
axes[5].set_yscale('log')
axes[5].grid(True)
axes[5].legend(loc='upper right')

# Plot ACE Bz (7th subplot) with color coding using LineCollection
points = np.array([hours_median, ace_bz_interp]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
colors = ['g' if (ace_bz_interp[i] + ace_bz_interp[i + 1]) / 2 >= 0 else 'r' for i in range(len(ace_bz_interp) - 1)]
lc = LineCollection(segments, colors=colors, linewidth=2, label='ACE Bz (GSM)')
axes[6].add_collection(lc)
axes[6].plot([], [], 'g-', label='ACE Bz (GSM) > 0')
axes[6].plot([], [], 'r-', label='ACE Bz (GSM) < 0')
axes[6].set_ylabel('Bz (nT)')
axes[6].grid(True)
axes[6].legend(loc='upper right')
axes[6].autoscale()

# Set x-axis properties
axes[6].set_xlabel('Time (UT Hours)')
axes[6].set_xticks(np.arange(0, 25, 1))
axes[6].set_xlim(0, 24)

# Adjust layout
plt.tight_layout()

# Save and close
plt.show()