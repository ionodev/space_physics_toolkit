import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Replace 'your_file.dat' with the actual filename
filepath = 'MSIS.dat'

# Define column names based on the header information in your file
column_names = ["Height", "O", "N2", "O2", "Mass_density", "Temperature_neutral"]

# Read the data into a Pandas DataFrame, skipping the header rows and specifying column names
df = pd.read_csv(filepath, skiprows=18, delim_whitespace=True, names=column_names)  # Use delim_whitespace for space-separated values

# Access data using column names
height = df["Height"] * 1e3 #m
height_km = df["Height"]
o_density = df["O"] * 1e6 #m-3
o2_density = df["O2"] * 1e6 #m-3
n2_density = df["N2"] * 1e6 #m-3
mass_density = df["Mass_density"] * 1e3 #kg m-3
temperature = df["Temperature_neutral"]

g_0 = 9.80665 #m/s^2 gravitational acceleration at Earth's surface
k = 1.380649e-23 #J/K boltzmann constant
amu = 1.660539e-27 #kg atomic mass unit
R_E = 6378e3 #m earth's radius

def H(T, m, g):
    # scale height
    # T : temperature
    # m : mass per molecule
    return k * T / (m*g)

def m(height_km):
    # mass per molecule
    N = o_density[height_km] + o2_density[height_km] + n2_density[height_km]  # total number density at height level
    rho = mass_density[height_km] # total mass density
    return rho/N #kg/molecule average mass per molecule

def T(height_km):
    return temperature[height_km] #K

def g(z):
    #z : height above ground in km
    return g_0 * pow(R_E/(R_E+z*1e3), 2)

# Task 1: scale height at ground level and 100km
H_0 = H(T(0), m(0), g_0)
H_100 = H(T(100), m(100), g_0)

m_O = 16 * amu
m_O2 = 32 * amu
m_N2 = 28 * amu

print(f'Scale height ground: {H_0*1e-3:.3g} km, Scale height 100 km: {H_100*1e-3:.3g} km')

H_O_120km = H(T(120), m_O, g_0)
H_O_600km = H(T(600), m_O, g_0)
H_O_120km_g = H(T(120), m_O, g(120))
H_O_600km_g = H(T(600), m_O, g(600))

H_O2_120km = H(T(120), m_O2, g_0)
H_O2_600km = H(T(600), m_O2, g_0)
H_O2_120km_g = H(T(120), m_O2, g(120))
H_O2_600km_g = H(T(600), m_O2, g(600))

H_N2_120km = H(T(120), m_N2, g_0)
H_N2_600km = H(T(600), m_N2, g_0)
H_N2_120km_g = H(T(120), m_N2, g(120))
H_N2_600km_g = H(T(600), m_N2, g(600))

def rel_error(approx, model):
    return 100*(approx-model)/model

print('\nScale height for g = 9.81 m/s^2 [km], Scale height for g(z) [km], error [%]')
print(f'H_O_120km = {H_O_120km*1e-3:.3g} km, H_O_120km_g = {H_O_120km_g*1e-3:.3g} km, error = {rel_error(H_O_120km, H_O_120km_g):.3g} %')
print(f'H_O_600km = {H_O_600km*1e-3:.3g} km, H_O_600km_g = {H_O_600km_g*1e-3:.3g} km, error = {rel_error(H_O_600km, H_O_600km_g):.3g} %')
print(f'H_O2_120km = {H_O2_120km*1e-3:.3g} km, H_O2_120km_g = {H_O2_120km_g*1e-3:.3g} km, error = {rel_error(H_O2_120km, H_O2_120km_g):.3g} %')
print(f'H_O2_600km = {H_O2_600km*1e-3:.3g} km, H_O2_600km_g = {H_O2_600km_g*1e-3:.3g} km, error = {rel_error(H_O2_600km, H_O2_600km_g):.3g} %')
print(f'H_N2_120km = {H_N2_120km*1e-3:.3g} km, H_N2_120km_g = {H_N2_120km_g*1e-3:.3g} km, error = {rel_error(H_N2_120km, H_N2_120km_g):.3g} %')
print(f'H_N2_600km = {H_N2_600km*1e-3:.3g} km, H_N2_600km_g = {H_N2_600km_g*1e-3:.3g} km, error = {rel_error(H_N2_600km, H_N2_600km_g):.3g} %')

def H_X(n_X, z):
    dnX_dz = np.gradient(n_X, z)
    return -1* n_X /(dnX_dz)

print('\nExperimental Scale Heights [km]:')
print(f'Hx_O_120km = {H_X(o_density, height)[120]*1e-3:.3g} km')
print(f'Hx_O_600km = {H_X(o_density, height)[600]*1e-3:.3g} km')
print(f'Hx_O2_120km = {H_X(o2_density, height)[120]*1e-3:.3g} km')
print(f'Hx_O2_600km = {H_X(o2_density, height)[600]*1e-3:.3g} km')
print(f'Hx_N2_120km = {H_X(n2_density, height)[120]*1e-3:.3g} km')
print(f'Hx_N2_600km = {H_X(n2_density, height)[600]*1e-3:.3g} km')

plt.plot(H(T(height_km), m_N2, g_0)*1e-3, height_km, label='N2')
plt.plot(H(T(height_km), m_O2, g_0)*1e-3, height_km, label='O2')
plt.plot(H(T(height_km), m_O, g_0)*1e-3, height_km, label='O')
plt.legend()
plt.title("Constant g")
plt.savefig('Const_g.pdf')
plt.show()

plt.plot(H(T(height_km), m_N2, g(height_km))*1e-3, height_km, label='N2')
plt.plot(H(T(height_km), m_O2, g(height_km))*1e-3, height_km, label='O2')
plt.plot(H(T(height_km), m_O, g(height_km))*1e-3, height_km, label='O')
plt.legend()
plt.title("Altitude varying g")
plt.savefig('Alt_var_g.pdf')
plt.show()

plt.plot(H(T(height_km), m_N2, g_0)*1e-3, height_km, label='N2 (g const.)')
plt.plot(H(T(height_km), m_N2, g(height_km))*1e-3, height_km, label='N2 (g variable)')
plt.plot(H_X(n2_density, height)[height_km]*1e-3, height_km, label='N2 (Exp.)')
plt.legend()
plt.savefig('H_N2.pdf')
plt.show()
plt.plot(H(T(height_km), m_O2, g_0)*1e-3, height_km, label='O2 (g const.)')
plt.plot(H(T(height_km), m_O2, g(height_km))*1e-3, height_km, label='O2 (g variable)')
plt.plot(H_X(o2_density, height)[height_km]*1e-3, height_km, label='O2 (Exp.)')
plt.legend()
plt.savefig('H_O2.pdf')
plt.show()
plt.plot(H(T(height_km), m_O, g_0)*1e-3, height_km, label='O (g const.)')
plt.plot(H(T(height_km), m_O, g(height_km))*1e-3, height_km, label='O (g variable)')
plt.plot((H_X(o_density, height)[height_km]*1e-3)[103:], height_km[103:], label='O (Exp.)')
plt.legend()
plt.savefig('H_O.pdf')
plt.show()

### plot error
plt.plot(height_km, rel_error(H(T(height_km), m_O2, g(height_km))*1e-3, H_X(o2_density, height)[height_km]*1e-3), label="O2 relative error [%]")
plt.plot(height_km, rel_error(H(T(height_km), m_N2, g(height_km))*1e-3, H_X(n2_density, height)[height_km]*1e-3), label="N2 relative error [%]")
plt.plot(height_km, rel_error((H(T(height_km), m_O, g(height_km))*1e-3), (H_X(o_density, height)[height_km]*1e-3)[103:]), label="O relative error [%]")
plt.legend()
plt.ylim(-100, 100)
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Variable g scale heights vs. Experimental scale heights")
plt.savefig('H_error_var_g_exp.pdf')
plt.show()

### plot error
plt.plot(height_km, rel_error(H(T(height_km), m_O2, g_0)*1e-3, H(T(height_km), m_O2, g(height_km))*1e-3) , label="O2 relative error [%]")
plt.plot(height_km, rel_error(H(T(height_km), m_N2, g_0)*1e-3, H(T(height_km), m_N2, g(height_km))*1e-3), label="N2 relative error [%]")
plt.plot(height_km, rel_error(H(T(height_km), m_O, g_0)*1e-3, H(T(height_km), m_O, g(height_km))*1e-3), label="O relative error [%]")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Constant g scale heights vs. Variable g scale heights")
plt.savefig('H_error_const_g_var_g.pdf')
plt.show()