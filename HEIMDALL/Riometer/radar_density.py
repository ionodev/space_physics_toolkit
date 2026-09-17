import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.colors as colors

# Load the data
file_path = "beata-20250303-20250304.mat"
mat_data = scipy.io.loadmat(file_path)
ne = mat_data.get('ne')  # electron density (39, 2869)
h = mat_data.get('h')    # altitude (39, 1)
T = mat_data.get('T')    # time (2869, 6)

# Convert time array to datetime
times = []
for t in T:
    year, month, day, hour, minute, second = t
    times.append(datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)))
times = np.array(times)

# Prepare data for plotting
h = h.flatten()  # Convert altitude to 1D array

# Filter altitudes up to 150 km
mask = h <= 150
h_filtered = h[mask]
ne_filtered = ne[mask, :]  # Select corresponding rows of electron density

# Clean the electron density data: remove invalid values and apply log scale
ne_filtered = np.where(ne_filtered > 0, ne_filtered, np.nan)  # Replace non-positive values with NaN
ne_filtered = np.where(np.isfinite(ne_filtered), ne_filtered, np.nan)  # Replace inf/-inf with NaN
ne_max = 3e12
ne_min = 3e9

# Create meshgrid for time and filtered altitude
X, Y = np.meshgrid(times, h_filtered)
Z = ne_filtered  # Filtered electron density

# Create the plot with a logarithmic color scale
plt.figure(figsize=(12, 8))
pcm = plt.pcolormesh(X, Y, Z, cmap='jet', norm=colors.LogNorm(vmin=ne_min, vmax=ne_max), shading='auto')
plt.colorbar(pcm, label='Electron Density (m^-3)')

# Format the x-axis with dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

# Set labels and title
plt.xlabel('Time (UT)')
plt.ylabel('Altitude (km)')
plt.title('EISCAT-UHF Electron Density (March 3-4, 2025)')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('eiscat_electron_density_uhf.png')