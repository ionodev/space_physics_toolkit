import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load ACE magnetometer data
ace_data = np.loadtxt('20250303_ace_mag_1m.txt', skiprows=14, usecols=(0, 1, 2, 3, 5, 6, 9), dtype=str)
print("ace_data shape:", ace_data.shape)
print("First 5 rows of ace_data:\n", ace_data[:5])

# Parse datetimes
ace_datetimes = [datetime.strptime(f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]}", "%Y %m %d %H%M") for row in ace_data]
ace_status = np.array([int(row[5]) for row in ace_data])
ace_bz = np.array([float(row[6]) for row in ace_data])

# Check status values
print("Unique status values:", np.unique(ace_status))

# Filter valid data (exclude Bz = -999.9)
valid_ace = (ace_bz != -999.9)
ace_datetimes = np.array(ace_datetimes)[valid_ace]
ace_bz = ace_bz[valid_ace]
ace_hours = np.array([(dt.hour + dt.minute / 60 + dt.second / 3600) for dt in ace_datetimes])
print("valid_ace count:", np.sum(valid_ace))
print("ace_hours length:", len(ace_hours))
print("ace_bz length:", len(ace_bz))
print("Sample valid data (first 5 points):")
print("Hours:", ace_hours[:5])
print("Bz:", ace_bz[:5])

# Create plot
fig, ax = plt.subplots(figsize=(12, 4))

# Plot Bz data
ax.plot(ace_hours, ace_bz, 'g-', label='ACE Bz (GSM)')
ax.set_title('ACE Magnetometer Bz Component (March 3, 2025 0 UT to March 4, 2025 0 UT)')

# Set axis properties
ax.set_xlabel('Time (UT Hours)')
ax.set_ylabel('Bz (nT)')
ax.set_xticks(np.arange(0, 25, 1))
ax.set_xlim(0, 24)
ax.grid(True)
ax.legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('plots/ace_bz_plot.png')
plt.show()
plt.close()