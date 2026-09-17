import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.colors as colors

# Load the data
file_path = "bella-20250303-20250304.mat"
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

# Filter altitudes between 90 and 150 km
mask = (h >= 90) & (h <= 150)
h_filtered = h[mask]
ne_filtered = ne[mask, :]  # Select corresponding rows of electron density

# Clean the electron density data: remove invalid values
ne_filtered = np.where(ne_filtered > 0, ne_filtered, np.nan)  # Replace non-positive values with NaN
ne_filtered = np.where(np.isfinite(ne_filtered), ne_filtered, np.nan)  # Replace inf/-inf with NaN

# Find median electron density for each time step
median_ne = np.nanmedian(ne_filtered, axis=0)  # median along altitude dimension

# Compute rolling median over 10 time samples
window_size = 15
rolling_median = np.full_like(median_ne, np.nan)  # Initialize array for rolling median
for i in range(window_size - 1, len(median_ne)):
    window = median_ne[i - window_size + 1:i + 1]
    rolling_median[i] = np.nanmedian(window)  # Compute median over the window

# Create the plot of rolling median electron density vs time
plt.figure(figsize=(12, 8))
plt.plot(times, rolling_median, 'b-', label=f'Median Electron Density (Rolling Window = {window_size})')

# Format the x-axis with dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

# Set labels and title
plt.xlabel('Time')
plt.ylabel('Median Electron Density (m^-3)')
plt.title('Rolling Median Electron Density (90-150 km) - March 3-4, 2025')
plt.yscale('log')
plt.grid(True)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.show()