import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

'''
Ring current is only on the around 20nA/m^2
'''

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability [Vs/Am]
R_E = 6.378 * 1e6  # Radius of Earth [m]

# Load the data
data = loadmat('noon-midnight-B-fields.mat')
BX = data['BX'] * 1e-9 # T
BXd = data['BXd'] * 1e-9 # T
BZ = data['BZ'] * 1e-9 # T
BZd = data['BZd'] * 1e-9 # T
Xnm = data['Xnm'] * R_E # m
Znm = data['Znm'] * R_E # m
Xre = data['Xnm']
Zre = data['Znm']
# 1. Remove the dipole component
BX_no_dipole = (BX-BXd) # T
BZ_no_dipole = (BZ-BZd) # T


# 2. Calculate the y-component of the curl
def curl_y(X, Z, BX, BZ):
    dX = int((X[0, 1] - X[0, 0])) # m
    dZ = int((Z[1, 0] - Z[0, 0])) # m

    print(dX)
    print(dZ)

    print("BX min/max:", np.min(BX_no_dipole)*1e9, np.max(BX_no_dipole)*1e9)
    print("BZ min/max:", np.min(BZ_no_dipole)*1e9, np.max(BZ_no_dipole)*1e9)
    print("Max absolute B difference in X direction:",
          1e9*np.max(np.abs(np.diff(BZ_no_dipole, axis=1))))
    print("Max absolute B difference in Z direction:",
          1e9*np.max(np.abs(np.diff(BX_no_dipole, axis=0))))

    dBZ_dX = np.gradient(BZ, dX, axis=1)
    dBX_dZ = np.gradient(BX, dZ, axis=0)

    return dBX_dZ - dBZ_dX


#Jy = curl_y(Xnm, Znm, BX_no_dipole, BZ_no_dipole) / mu_0
Jy = curl_y(Xnm, Znm, BX_no_dipole, BZ_no_dipole) / mu_0


# Convert current density to nA/m^2
Jy_nA = Jy * 1e9 *10 # nA/m^2 (here I've applied a correction by a factor of 10?)

# Compute radial distance from Earth in units of R_E
#R = np.sqrt(Xnm**2 + Znm**2)

# Flip the sign of Jy where r < 2 R_E
#mask_inner_region = R < 1.2  # radius in units of Earth radii
#Jy_nA[mask_inner_region] *= -1


# 3. Display the current using pcolormesh (in nA/m^2)
plt.figure(figsize=(8, 6))
# Use a diverging colormap with white around zero
cmap_jy = plt.get_cmap('RdBu_r')  # 'RdBu_r' reverses the colormap so red is positive and blue is negative, and it has white in the middle

# Find the maximum and minimum values of the current density for setting symmetric limits
max_Jy = np.max(Jy_nA)
min_Jy = np.min(Jy_nA)
max_abs_Jy = np.max(np.abs(Jy_nA))  # Find the maximum absolute value

print(max_Jy)
print(min_Jy)

# Set the color scale limits
# Determine the limits to center the colormap around zero
# vmax = np.max([max_abs_Jy, 30])  # Use max_abs_Jy or 30, whichever is greater
# vmin = -vmax

max_abs_Jy = np.max(np.abs(Jy_nA))
norm_jy = plt.Normalize(vmin=-100, vmax=+100)


# Create the pcolormesh plot and store the returned object
mesh = plt.pcolormesh(Xre, Zre, Jy_nA, cmap=cmap_jy, norm=norm_jy, shading='auto', rasterized=True)

# Create the colorbar using the mesh object and the current axes
cbar = plt.gcf().colorbar(mesh, label='Current Density (nA/m^2)')

plt.xlabel('X (Re)')
plt.ylabel('Z (Re)')
plt.title('Magnetospheric Current Density (Jy)')
ax = plt.gca()
ax.invert_xaxis()

# Adding the labels to the plot
labels = {
    (12, 0): "Dayside Magnetopause Current",
    (5.5, 0): "Ring-current (westward)",
    (-5.5, 0): "Ring-current (eastward)",
    (-30, 0): "Neutral Sheet Current",
    (-30, 24): "Tail current (northern tail)",
    (-30, -24): "Tail current (southern tail)"
}

# Place the labels
# Place the labels
for (x, z), label in labels.items():
    if label == "Dayside Magnetopause Current":
        plt.text(x, z, label, fontsize=5, ha='center', va='center', color='black', fontweight='bold', rotation=90)
    else:
        plt.text(x, z, label, fontsize=5, ha='center', color='black', fontweight='bold')


plt.show()



# 4. Plot Jy with the magnetic field arrows (quiver) (in nA/m^2)
plt.figure(figsize=(8, 6))

# Use the same colormap and limits for consistency
mesh = plt.pcolormesh(Xre, Zre, Jy_nA, cmap=cmap_jy, norm=norm_jy, shading='auto', rasterized=True)
cbar = plt.gcf().colorbar(mesh, label='Current Density (nA/m^2)')
plt.xlabel('X (Re)')
plt.ylabel('Z (Re)')
plt.title('Magnetospheric Current Density (Jy) with Normalized B-field')

# Calculate a suitable spacing for the arrows
arrow_spacing = 15  # Adjust as needed for clarity
X_quiver, Z_quiver = np.meshgrid(Xre[0, ::arrow_spacing], Zre[::arrow_spacing, 0])
BX_quiver = BX[::arrow_spacing, ::arrow_spacing] * 1e9  # Convert to nT
BZ_quiver = BZ[::arrow_spacing, ::arrow_spacing] * 1e9  # Convert to nT

# Calculate the magnitude of the B-field vectors
B_magnitude = np.sqrt(BX_quiver**2 + BZ_quiver**2)

# Avoid division by zero
B_magnitude = np.where(B_magnitude == 0, 1e-8, B_magnitude)

# Apply logarithmic normalization but set the minimum value to 1 for the log scale
log_norm = mcolors.LogNorm(vmin=1, vmax=np.max(B_magnitude))  # Set minimum value to avoid log(0)

# Normalize vectors
BX_quiver_norm = -BX_quiver / B_magnitude  # Reverse the X-direction
BZ_quiver_norm = BZ_quiver / B_magnitude

# Create the colormap and include black at the bottom
jet_cmap = cm.get_cmap('jet', 256)

# Add black to the beginning of the list (this is done before creating the custom colormap)
black_jet_colors = [(0, 0, 0)] + [jet_cmap(i) for i in range(jet_cmap.N)]

# Create the custom colormap with black at the bottom
black_jet_cmap = mcolors.LinearSegmentedColormap.from_list("black_jet", black_jet_colors, N=257)

# Use quiver with color representing magnitude
ax_quiver = plt.gca()  # Get the current axes
quiver = ax_quiver.quiver(X_quiver,
                           Z_quiver,
                           BX_quiver_norm,
                           BZ_quiver_norm,
                           B_magnitude,  # Pass magnitude for the logarithmic scale
                           cmap=black_jet_cmap,  # Use the custom colormap with black at the bottom
                           norm=log_norm,  # Apply logarithmic normalization
                           alpha=0.7,
                           scale=50)  # Adjust this value to control the arrow size

# Add a colorbar for the quiver plot
sm = plt.cm.ScalarMappable(cmap=black_jet_cmap, norm=log_norm)
sm.set_array([])  # For older versions of matplotlib
cbar_quiver = plt.colorbar(sm, ax=ax_quiver, label='Magnetic Field Magnitude (nT)')

# Invert the X-axis if needed (for consistency with previous plots)
ax_quiver.invert_xaxis()

# Show the plot
plt.show()


# Find the index where Z is closest to zero
z_index = np.argmin(np.abs(Znm[:, 0]))  # Find the row corresponding to Z ≈ 0

# Extract the Jy values and corresponding X positions along Z = 0
Jy_equator = Jy_nA[z_index, :]  # Current density at Z = 0
X_line = Xre[z_index, :]        # Corresponding X positions in Re

# Plot
plt.figure(figsize=(8, 5))
plt.plot(X_line, Jy_equator, label=r'$J_y$ along $Z=0$')
plt.yticks(np.arange(-225, 225 + 25, 25))
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel('X (Re)')
plt.ylabel(r'$J_y$ (nA/m²)')
plt.title(r'Equatorial Current Density $J_y$ along $Z=0$')
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()  # Optional: if you want tail to the left
plt.show()

# 1. Plot Jy with the magnetic field lines (streamplot) (in nA/m^2)
plt.figure(figsize=(8, 6))

# Use the same colormap and limits for consistency for current density
mesh = plt.pcolormesh(Xre, Zre, Jy_nA, cmap=cmap_jy, norm=norm_jy, shading='auto', rasterized=True)
cbar = plt.gcf().colorbar(mesh, label='Current Density (nA/m^2)')
plt.xlabel('X (Re)')
plt.ylabel('Z (Re)')
plt.title('Magnetospheric Current Density (Jy) with Magnetic Field Lines')
ax = plt.gca()
ax.invert_xaxis()

# Streamplot requires U and V components for the vector field
U = BX  # X-component of the magnetic field
V = BZ  # Z-component of the magnetic field

# Create the streamplot to plot magnetic field lines
plt.streamplot(Xre, Zre, U, V, color="k", linewidth=0.5, cmap='jet', density=4)

# Show the plot
plt.show()