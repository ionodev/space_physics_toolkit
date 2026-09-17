import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot parameters
a4_width = 8.27
a4_height = 11.69
plt.rcParams.update({'figure.figsize': (a4_width, a4_height/2), 'font.size': 16, 'font.family': 'Times New Roman'})

filepath = 'MSIS.dat'

# Define column names based on the header information in your file
column_names = ["Height", "O", "N2", "O2", "Mass_density", "Temperature_neutral"]

# Read the data into a Pandas DataFrame, skipping the header rows and specifying column names
df = pd.read_csv(filepath, skiprows=18, delim_whitespace=True, names=column_names)  # Use delim_whitespace for space-separated values

# Access data using column names
height = df["Height"] * 1e3 #m
o_density = df["O"] * 1e6 #m-3
o2_density = df["O2"] * 1e6 #m-3
n2_density = df["N2"] * 1e6 #m-3
mass_density = df["Mass_density"] * 1e3 #kg m-3
temperature = df["Temperature_neutral"]

g_approx = 9.28 #m/s^2 approximate gravitational constant
g_0 = 9.80 #m/s^2 gravitational acceleration at Earth's surface
k = 1.380649e-23 #J/K boltzmann constant
amu = 1.660539e-27 #kg atomic mass unit
R_E = 6378e3 #m earth's radius
atm = 101325 #Pa atmosphere unit

def rho(z):
    index = z*1e-3
    return mass_density[index]

def n(z):
    index = z*1e-3
    n_tot = o_density[index] + o2_density[index] + n2_density[index]
    return n_tot

def m(z):
    return rho(z)/n(z)

def T(z):
    index = z*1e-3
    return temperature[index] #K

def g(z):
    #z : height above ground in m
    return g_0 * pow(R_E/(R_E+z), 2)

def H(z):
    # scale height
    # T : temperature
    # m : mass per molecule
    return k * T(z) / (m(z)*g(z))
def P_idg(z):
    return n(z)*k*T(z)

P_0 = P_idg(0) # ground-level pressure assumed to follow ideal gas law

def P(z_values):
    """
    Calculates the pressure profile P(z) given altitude-dependent scale height.

    Args:
        z_values: Array of altitude values.
        H_values: Array of scale height values corresponding to the altitudes.
        P_0: Pressure at sea level (z=0).

    Returns:
        Array of pressure values corresponding to the given altitudes.
    """

    # Ensure arrays are NumPy arrays for efficient calculations
    z_values = np.asarray(z_values)
    H_values = np.asarray(H(z_values))


    # Calculate 1/H(z)
    inv_H = 1.0 / H_values

    # Calculate the integral using the trapezoidal rule with np.trapz()
    integral_trapezoidal = np.zeros_like(z_values)  # Initialize array for integral values
    for i in range(len(z_values)):
        integral_trapezoidal[i] = np.trapz(inv_H[:i + 1], z_values[:i + 1])  # Definite integral up to z[i]

    # Calculate pressure profile using both methods
    P_z_trapezoidal = P_0 * np.exp(-integral_trapezoidal)

    return P_z_trapezoidal

def rel_error(approx, model):
    return 100*(approx-model)/model

z_values = np.arange(0, 600e3 + 1e3, 1e3)
P_values = P(z_values)


plt.plot(z_values*1e-3, P_values/atm, label='Atmospheric Pressure', c='blue')
#plt.savefig('atmospheric_pressure.pdf')

plt.plot(z_values*1e-3, P_idg(z_values)/atm, label='Pressure (Ideal Gas Law)', c='red')
plt.legend()
plt.xlabel('Altitude [km]')
plt.ylabel('Pressure [atm]')
plt.yscale('log')
plt.tight_layout
#plt.savefig('pressure_idg.pdf')
plt.show()

plt.plot(z_values*1e-3, rel_error(P_idg(z_values)/atm, P_values/atm), label='Relative error [%]', c='black')
plt.legend()
plt.xlabel('Altitude [km]')
plt.ylabel('Error [%]')
plt.tight_layout
#plt.savefig('error.pdf')
plt.show()

alpha = 9.8e-3 #K/m adiabatic lapse rate
dT_dz = np.gradient(T(z_values), z_values, edge_order=2)
unstable_regions = dT_dz < -1*abs(alpha)
plt.plot(T(z_values),z_values*1e-3)
#plt.plot(dT_dz*z_values,z_values*1e-3, label='reconstruct')
plt.xlabel('Temperature [K]')
plt.ylabel('Altitude [km]')
#plt.savefig('temperature.pdf')
plt.show()
plt.plot(dT_dz*1e3, z_values*1e-3, )
plt.ylabel('Altitude [km]')
plt.xlabel('K/km')
#plt.savefig('adiabatic_lapse_rate.pdf')
plt.show()

print(height[unstable_regions])