import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Helper function to add separator page ---
def add_task_separator_page(pdf_pages, task_number):
    """Creates a blank page with 'Task X' text and adds it to the PDF."""
    fig = plt.figure(figsize=(8.27, 11.69)) # A4 size in inches
    plt.axis('off') # Turn off axes
    plt.text(0.5, 0.5, f'Task {task_number}',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=20,
             transform=fig.transFigure)
    pdf_pages.savefig(fig)
    plt.close(fig)

height = 110e3

# Load the data
MSIS = np.loadtxt('MSIS.dat', skiprows=18)
IRI = np.loadtxt('IRI.dat', skiprows=46)

# Extract columns
altitude_MSIS = MSIS[100:, 0] * 1e3  # Height (m)
altitude_IRI = IRI[:, 0] * 1e3 # Height (m)
Tn_interp = interp1d(altitude_MSIS, MSIS[100:, 5], fill_value="extrapolate") # Neutral temperature (K) function
Tn = Tn_interp(altitude_IRI) # Tn interpolated to IRI altitudes

dt = 1 # s (time which corresponds to one index)
# Integration time for continuity equations
time_0 = np.arange(0, 3600 + dt, dt) # needs to be changed for each task
time_1 = np.arange(0, 600 + dt, dt)

# Temperatures (align with altitude_IRI)
Ti = IRI[:, 2] # Ion temperature (K)
Te = IRI[:, 3] # Electron temperature (K)

def initial_state(altitude): # "initial state" used for calculating improved initial states in task 0
    # Find the index in altitude_IRI closest to the requested altitude
    j = np.argmin(np.abs(altitude_IRI - altitude))
    n_i_data = IRI[:, 1]  # Ion number density (m^-3) (assumes quasi-neutral plasma)
    n_e_init = n_i_data[j]  # Electron number density (m^-3)
    n_O_i_init = ((IRI[:, 4] * n_i_data) * 0.01)[j]  # O+ density (m^-3)
    n_O2_i_init = ((IRI[:, 7] * n_i_data) * 0.01)[j]  # O2+ density (m^-3)
    n_NO_i_init = ((IRI[:, 8] * n_i_data) * 0.01)[j]  # NO+ density (m^-3)
    n_N2_i_init = 1e3  # N2+ density (m^-3) (not included in data, assume constant initial)
    n_NO_init = 1e3  # NO density (m^-3) (not included in data, assume constant initial)
    n_state_init = [0,0,0,0,0,0]#[n_O_i_init, n_O2_i_init, n_N2_i_init, n_NO_i_init, n_NO_init, n_e_init]
    return n_state_init

def n_major(time):
    # Interpolate MSIS data to IRI altitudes first
    n_O_interp = interp1d(altitude_MSIS, MSIS[100:, 1] * 1e6, fill_value="extrapolate")
    n_N2_interp = interp1d(altitude_MSIS, MSIS[100:, 2] * 1e6, fill_value="extrapolate")
    n_O2_interp = interp1d(altitude_MSIS, MSIS[100:, 3] * 1e6, fill_value="extrapolate")
    n_O_iri_alt = n_O_interp(altitude_IRI)
    n_N2_iri_alt = n_N2_interp(altitude_IRI)
    n_O2_iri_alt = n_O2_interp(altitude_IRI)
    # Tile for the time dimension
    n_O = np.tile(n_O_iri_alt, (len(time), 1)).T # O density (m^-3)
    n_N2 = np.tile(n_N2_iri_alt, (len(time), 1)).T # N2 density (m^-3)
    n_O2 = np.tile(n_O2_iri_alt, (len(time), 1)).T # O2 density (m^-3)
    return [n_O, n_O2, n_N2]

def calculate_rates(Ti, Te, altitude, time):
    # Use Tn, Ti, Te as provided
    Tn_adj = Tn
    Ti_adj = Ti
    Te_adj = Te
    Tr = (Tn_adj + Ti_adj) / 2  # Neutral-ion temperature average (K)
    # Ensures temperatures have the same shape as altitude for tiling
    Te_reshaped = Te_adj[:, np.newaxis]
    Tr_reshaped = Tr[:, np.newaxis]
    altitude_reshaped = altitude[:, np.newaxis]
    # Reaction rates (m^3/s) (table 4.4, Brekke, with adjustments)
    # Dissociative recombination
    alpha_1 = np.tile(2.1e-13 * (Te_reshaped/300)**(-0.85), (1, len(time)))
    alpha_2 = np.tile(1.9e-13 * (Te_reshaped/300)**(-0.5), (1, len(time)))
    alpha_3 = np.tile(1.8e-13 * (Te_reshaped/300)**(-0.39), (1, len(time)))
    # Radiative recombination
    alpha_r = np.tile(3.7e-18 * (250 / Te_reshaped) ** (0.7), (1, len(time)))
    # Rearrangement
    k1 = np.tile(2e-18 * np.ones_like(altitude_reshaped), (1, len(time)))
    k2 = np.tile(2e-17 * (Tr_reshaped/300)**(-0.4), (1, len(time)))
    k3 = np.tile(4.4e-16 * np.ones_like(altitude_reshaped), (1, len(time)))
    k4 = np.tile(5e-22 * np.ones_like(altitude_reshaped), (1, len(time)))
    k5 = np.tile(1.4e-16 * (Tr_reshaped/300)**(-0.44), (1, len(time)))
    k6 = np.tile(5e-17 * (Tr_reshaped/300)**(-0.8), (1, len(time)))
    k_values = [k1, k2, k3, k4, k5, k6]
    alpha_values = [alpha_1, alpha_2, alpha_3, alpha_r]
    return alpha_values, k_values

def q_major_i(q_e, n_major):
    n_O, n_O2, n_N2 = n_major
    denominator = (0.92*n_N2 + n_O2 + 0.56 * n_O)
    q_N2_i = q_e * (0.92 * n_N2)/denominator
    q_O2_i = q_e * n_O2 / denominator
    q_O_i = q_e * (0.56 * n_O) / denominator
    return [q_O_i, q_O2_i, q_N2_i, q_e]

def continuity_equations(t, n_state, n_major_interp, q_values_interp, k_values_interp, alpha_values_interp, altitude, altitude_grid, return_terms=False):
    n_O_i, n_O2_i, n_N2_i, n_NO_i, n_NO, n_e = n_state
    # Find the index in the altitude_grid corresponding to the target altitude
    j = np.argmin(np.abs(altitude_grid - altitude))
    # Interpolate to the current time t and get value at index j
    n_O = n_major_interp[0](t)[j]
    n_O2 = n_major_interp[1](t)[j]
    n_N2 = n_major_interp[2](t)[j]
    q_O_i = q_values_interp[0](t)[j]
    q_O2_i = q_values_interp[1](t)[j]
    q_N2_i = q_values_interp[2](t)[j]
    q_e = q_values_interp[3](t)[j]
    k1, k2, k3, k4, k5, k6 = [k(t)[j] for k in k_values_interp]
    alpha_1, alpha_2, alpha_3, alpha_r = [a(t)[j] for a in alpha_values_interp]
    # Source and loss terms for each species
    # O+
    P_O_i_q = q_O_i
    L_O_i_k1 = k1 * n_N2 * n_O_i
    L_O_i_k2 = k2 * n_O2 * n_O_i
    L_O_i_alphar = alpha_r * n_O_i * n_e
    P_O_i = P_O_i_q
    L_O_i = L_O_i_k1 + L_O_i_k2 + L_O_i_alphar
    # O2+
    P_O2_i_q = q_O2_i
    P_O2_i_k2 = k2 * n_O2 * n_O_i
    P_O2_i_k6 = k6 * n_O2 * n_N2_i
    L_O2_i_k3 = k3 * n_O2_i * n_NO
    L_O2_i_k4 = k4 * n_N2 * n_O2_i
    L_O2_i_alpha2 = alpha_2 * n_O2_i * n_e
    P_O2_i = P_O2_i_q + P_O2_i_k2 + P_O2_i_k6
    L_O2_i = L_O2_i_k3 + L_O2_i_k4 + L_O2_i_alpha2
    # N2+
    P_N2_i_q = q_N2_i
    L_N2_i_k5 = k5 * n_O * n_N2_i
    L_N2_i_k6 = k6 * n_O2 * n_N2_i
    L_N2_i_alpha3 = alpha_3 * n_N2_i * n_e
    P_N2_i = P_N2_i_q
    L_N2_i = L_N2_i_k5 + L_N2_i_k6 + L_N2_i_alpha3
    # NO+
    # no main prod term
    P_NO_i_k1 = k1 * n_N2 * n_O_i
    P_NO_i_k3 = k3 * n_O2_i * n_NO
    P_NO_i_k4 = k4 * n_N2 * n_O2_i
    P_NO_i_k5 = k5 * n_O * n_N2_i
    L_NO_i_alpha1 = alpha_1 * n_NO_i * n_e
    P_NO_i = P_NO_i_k1 + P_NO_i_k3 + P_NO_i_k4 + P_NO_i_k5
    L_NO_i = L_NO_i_alpha1
    # NO (neutral)
    P_NO_k4 = k4 * n_N2 * n_O2_i
    L_NO_k3 = k3 * n_O2_i * n_NO
    P_NO = P_NO_k4
    L_NO = L_NO_k3
    # Electrons
    P_e_q = q_e # production of electrons
    L_e_alpha1 = alpha_1 * n_NO_i * n_e
    L_e_alpha2 = alpha_2 * n_O2_i * n_e
    L_e_alpha3 = alpha_3 * n_N2_i * n_e
    L_e_alphar = alpha_r * n_O_i * n_e
    P_e = P_e_q
    L_e = L_e_alpha1 + L_e_alpha2 + L_e_alpha3 + L_e_alphar
    # Derivatives
    dn_O_i_dt = P_O_i - L_O_i
    dn_O2_i_dt = P_O2_i - L_O2_i
    dn_N2_i_dt = P_N2_i - L_N2_i
    dn_NO_i_dt = P_NO_i - L_NO_i
    dn_NO_dt = P_NO - L_NO
    dn_e_dt = P_e - L_e # Use calculated electron rate
    dn_dt = np.array([dn_O_i_dt, dn_O2_i_dt, dn_N2_i_dt, dn_NO_i_dt, dn_NO_dt, dn_e_dt])
    if return_terms:
        source_terms = {
            'O+': [P_O_i_q],
            'O2+': [P_O2_i_q, P_O2_i_k2, P_O2_i_k6],
            'N2+': [P_N2_i_q],
            'NO+': [P_NO_i_k1, P_NO_i_k3, P_NO_i_k4, P_NO_i_k5],
            'NO': [P_NO_k4],
            'e-': [P_e_q],
        }
        loss_terms = {
            'O+': [L_O_i_k1, L_O_i_k2, L_O_i_alphar],
            'O2+': [L_O2_i_k3, L_O2_i_k4, L_O2_i_alpha2],
            'N2+': [L_N2_i_k5, L_N2_i_k6, L_N2_i_alpha3],
            'NO+': [L_NO_i_alpha1],
            'NO': [L_NO_k3],
            'e-': [L_e_alpha1, L_e_alpha2, L_e_alpha3, L_e_alphar],
        }
        return dn_dt, source_terms, loss_terms
    else:
        return dn_dt

def solve_ion_chem(time, n_major_data, q_values_data, k_values_data, alpha_values_data, altitude_target, y_0, altitude_grid):
    # Create interpolation functions (ensure data covers the time range)
    # Use bounds_error=False and fill_value="extrapolate" if time might go slightly out of bounds
    interp_options = {'axis': 1, 'kind': 'linear', 'bounds_error': False, 'fill_value': 'extrapolate'}
    n_major_interp = [interp1d(time, n_major_data[i], **interp_options) for i in range(3)]
    q_values_interp = [interp1d(time, q_values_data[i], **interp_options) for i in range(4)]
    k_values_interp = [interp1d(time, k, **interp_options) for k in k_values_data]
    alpha_values_interp = [interp1d(time, a, **interp_options) for a in alpha_values_data]
    # Time span for integration
    t_span = (time[0], time[-1])
    t_eval = time  # Evaluate at the pre-defined time points
    # Solve the ODE system
    sol = solve_ivp(
        fun=lambda t, n_state: continuity_equations(t, n_state, n_major_interp, q_values_interp, k_values_interp, alpha_values_interp, altitude_target, altitude_grid, return_terms=False),
        t_span=t_span,
        y0=y_0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-6, atol=1e-8 # Adjust tolerances if needed
    )
    # Arrays to store *individual* source and loss terms
    source_terms_all = {'O+': [], 'O2+': [], 'N2+': [], 'NO+': [], 'NO': [], 'e-': []}
    loss_terms_all = {'O+': [], 'O2+': [], 'N2+': [], 'NO+': [], 'NO': [], 'e-': []}
    # Recalculate terms using the solution
    for i, t in enumerate(sol.t):
        n_state = sol.y[:, i]
        t_interp = t # Ensure t is within bounds for interpolation
        _, source_terms, loss_terms = continuity_equations(t_interp, n_state, n_major_interp, q_values_interp, k_values_interp, alpha_values_interp, altitude_target, altitude_grid, return_terms=True)
        for species in source_terms_all:
            source_terms_all[species].append(source_terms[species])
        for species in loss_terms_all:
            loss_terms_all[species].append(loss_terms[species])
    # Convert lists of lists to numpy arrays
    for species in source_terms_all:
        max_len = max(len(lst) for lst in source_terms_all[species])
        padded_list = [lst + [0.0] * (max_len - len(lst)) for lst in source_terms_all[species]]
        source_terms_all[species] = np.array(padded_list).T
    for species in loss_terms_all:
        max_len = max(len(lst) for lst in loss_terms_all[species])
        padded_list = [lst + [0.0] * (max_len - len(lst)) for lst in loss_terms_all[species]]
        loss_terms_all[species] = np.array(padded_list).T
    return sol, source_terms_all, loss_terms_all, altitude_target

def make_plots(sol_data, pdf_pages):
    """Generates and saves plots for a given simulation result."""
    n, source_terms_all, loss_terms_all, height = sol_data
    time = n.t
    species_labels = ['O+', 'O2+', 'N2+', 'NO+', 'NO', 'e-']
    # Define labels
    reaction_labels = {
        'O+': {
            'source': [r'$q_{O^+}$'],
            'loss': [r'$k_1 n_{N_2} n_{O^+}$', r'$k_2 n_{O_2} n_{O^+}$', r'$\alpha_r n_{O^+} n_e$']
        },
        'O2+': {
            'source': [r'$q_{O_2^+}$', r'$k_2 n_{O_2} n_{O^+}$', r'$k_6 n_{O_2} n_{N_2^+}$'],
            'loss': [r'$k_3 n_{O_2^+} n_{NO}$', r'$k_4 n_{N_2} n_{O_2^+}$', r'$\alpha_2 n_{O_2^+} n_e$']
        },
        'N2+': {
            'source': [r'$q_{N_2^+}$'],
            'loss': [r'$k_5 n_O n_{N2^+}$', r'$k_6 n_{O_2} n_{N_2^+}$', r'$\alpha_3 n_{N_2^+} n_e$']
        },
        'NO+': {
            'source': [r'$k_1 n_{N_2} n_{O^+}$', r'$k_3 n_{O_2^+} n_{NO}$', r'$k_4 n_{N_2} n_{O_2^+}$', r'$k_5 n_O n_{N_2^+}$'],
            'loss': [r'$\alpha_1 n_{NO^+} n_e$']
        },
        'NO': {
             'source': [r'$k_4 n_{N_2} n_{O_2^+}$'],
             'loss': [r'$k_3 n_{O_2^+} n_{NO}$']
        },
        'e-': {
            'source': [r'$q_e$'],
            'loss': [r'$\alpha_1 n_{NO^+} n_e$', r'$\alpha_2 n_{O_2^+} n_e$', r'$\alpha_3 n_{N_2^+} n_e$', r'$\alpha_r n_{O^+} n_e$']
        }
    }
    # --- 1. Densities vs. Time ---
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(species_labels)):
        plt.plot(time, n.y[i], label=species_labels[i])
    plt.xlabel('Time [s]')
    plt.ylabel(r'Density [m$^{-3}$]')
    plt.title(f'Densities at {height / 1e3:.0f} km')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e1, 1e12)
    plt.yticks([10 ** i for i in range(1, 13)])
    #plt.ylim(bottom=1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    pdf_pages.savefig(fig)
    plt.close(fig)
    # --- 2. Total Source vs. Total Loss for Each Species ---
    for i, species in enumerate(species_labels):
        fig = plt.figure(figsize=(12, 6))
        total_source = np.sum(source_terms_all[species], axis=0)
        total_loss = np.sum(loss_terms_all[species], axis=0)
        plt.plot(time, total_source, label=f'Total Source ({species})')
        plt.plot(time, total_loss, label=f'Total Loss ({species})')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Rate [m$^{-3}$ s$^{-1}$]')
        plt.title(f'Total Source/Loss for {species} at {height / 1e3:.0f} km')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        pdf_pages.savefig(fig)
        plt.close(fig)
    # --- 3 & 4. Charge Conservation ---
    fig = plt.figure(figsize=(12, 6))
    n_i_total = np.sum(n.y[:4], axis=0) # Sum O+, O2+, N2+, NO+
    n_e_total = n.y[5] # Electron density is the 6th element (index 5)
    plt.plot(time, n_i_total, label='Total Ion Density')
    plt.plot(time, n_e_total, label='Electron Density', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel(r'Density [m$^{-3}$]')
    plt.title(f'Charge Conservation Check at {height / 1e3:.0f} km')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    pdf_pages.savefig(fig)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 6))
    ratio = n_i_total / n_e_total
    plt.plot(time, ratio, label='Total Ion / Electron Ratio')
    plt.xlabel('Time [s]')
    plt.ylabel('Ratio')
    plt.title(f'Ion-to-Electron Ratio at {height / 1e3:.0f} km')
    plt.ylim(0, 2)  # Adjust Y-axis limits if needed
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    pdf_pages.savefig(fig)
    plt.close(fig)
    # --- 5. Individual Reaction Rates (Stacked Area Plots) ---
    for species in species_labels:
        source_data = source_terms_all[species]
        source_labels = reaction_labels[species]['source']
        if source_data.shape[0] == len(source_labels):
            fig = plt.figure(figsize=(12, 6))
            plt.stackplot(time, source_data, labels=source_labels)
            plt.xlabel('Time [s]')
            plt.ylabel(r'Source Rate [m$^{-3}$ s$^{-1}$]')
            plt.title(f'Individual Source Terms for {species} at {height/1e3:.0f} km')
            plt.legend(loc='upper left', fontsize='small')
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            pdf_pages.savefig(fig)
            plt.close(fig)
        loss_data = loss_terms_all[species]
        loss_labels = reaction_labels[species]['loss']
        if loss_data.shape[0] == len(loss_labels):
            fig = plt.figure(figsize=(12, 6))
            plt.stackplot(time, loss_data, labels=loss_labels)
            plt.xlabel('Time [s]')
            plt.ylabel(r'Loss Rate [m$^{-3}$ s$^{-1}$]')
            plt.title(f'Individual Loss Terms for {species} at {height/1e3:.0f} km')
            plt.legend(loc='upper left', fontsize='small')
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            pdf_pages.savefig(fig)
            plt.close(fig)

# --- Task 3 Specific Functions ---
def task_3_analysis(sol_result, time, n_major_off, alpha_values_off, k_values_off, altitude):
    """Analyzes the electron density decay after ionization shutoff."""
    sol = sol_result[0] # Extract the solution object
    time_off_index = np.argmin(np.abs(time - 100))  # Index closest to 100 seconds
    n_e_0 = sol.y[5, time_off_index]  # n_e at shutoff (index 5)
    n_NO_plus_off = sol.y[3, time_off_index] # index 3
    n_O2_plus_off = sol.y[1, time_off_index] # index 1
    n_N2_plus_off = sol.y[2, time_off_index] # index 2
    alpha_1, alpha_2, alpha_3, _ = alpha_values_off # Get alpha values at shutoff
    k1, k2, _, _, _, _ = k_values_off # Get k values at shutoff
    n_N2 = n_major_off[2] # N2 density at shutoff
    n_O2 = n_major_off[1] # O2 density at shutoff
    r_1 = n_NO_plus_off / n_e_0
    r_2 = n_O2_plus_off / n_e_0
    r_3 = n_N2_plus_off / n_e_0
    alpha_eff = alpha_1 * r_1 + alpha_2 * r_2 + alpha_3 * r_3 # Brekke 4.32 p. 224 (Brekke neglects radiative recombination)
    beta_denom = (1 + (k1 / alpha_1) * (n_N2 / n_e_0) + (k2 / alpha_2) * (n_O2 / n_e_0)) # Brekke p. 212
    beta = (k1 * n_N2 + k2 * n_O2) / beta_denom # Brekke p. 212
    n_e_recomb = np.zeros_like(time, dtype=float)
    n_e_beta = np.zeros_like(time, dtype=float)
    time_after_shutoff = time - 100.0
    # Calculate decay only for times >= 100s
    mask = time >= 100.0
    n_e_recomb[mask] = n_e_0 / (1 + alpha_eff * n_e_0 * time_after_shutoff[mask])
    n_e_beta[mask] = n_e_0 * np.exp(-beta * time_after_shutoff[mask])
    # For times < 100s, set to NaN
    n_e_recomb[~mask] = np.nan
    n_e_beta[~mask] = np.nan
    return n_e_recomb, n_e_beta, beta

def make_task3_plots(pdf_pages, time_1, sol_1_110km_res, sol_1_230km_res,
                     n_e_recomb_110, n_e_beta_110, beta_110,
                     n_e_recomb_230, n_e_beta_230, beta_230):
    """Generates and saves plots for Task 3."""
    # Create a mask for time >= 100 seconds for plotting decay
    time_plot_mask = time_1 >= 100
    time_plot = time_1 # Plot full simulation time
    # --- 110 km ---
    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_plot, sol_1_110km_res[0].y[5], label='Simulation $n_e$') # Index 5 is electron density
    plt.plot(time_1[time_plot_mask], n_e_recomb_110[time_plot_mask], label=r'$\alpha$-Decay Approx.', linestyle='--')
    plt.plot(time_1[time_plot_mask], n_e_beta_110[time_plot_mask], label=fr'$\beta$-Decay Approx. ($\beta={beta_110:.2e}$ s$^{{-1}}$)', linestyle=':') # Use scientific notation for beta
    plt.xlabel('Time [s]')
    plt.ylabel(r'Electron Density [m$^{-3}$]')
    plt.title('Electron Density Decay Comparison at 110 km')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e9, 1e12)
    plt.yticks([10 ** i for i in range(9, 13)])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    pdf_pages.savefig(fig)
    plt.close(fig)
    # --- 230 km ---
    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_plot, sol_1_230km_res[0].y[5], label='Simulation $n_e$') # Index 5 is electron density
    plt.plot(time_1[time_plot_mask], n_e_recomb_230[time_plot_mask], label=r'$\alpha$-Decay Approx.', linestyle='--')
    plt.plot(time_1[time_plot_mask], n_e_beta_230[time_plot_mask], label=fr'$\beta$-Decay Approx. ($\beta={beta_230:.2e}$ s$^{{-1}}$)', linestyle=':')
    plt.xlabel('Time [s]')
    plt.ylabel(r'Electron Density [m$^{-3}$]')
    plt.title('Electron Density Decay Comparison at 230 km')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e9, 1e12)
    plt.yticks([10 ** i for i in range(9, 13)])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    pdf_pages.savefig(fig)
    plt.close(fig)

# --- Task 4 Specific Functions ---
def q_e_4(time, altitude_grid):
    # Ensure time is a numpy array
    time = np.asarray(time)
    # Create the time-varying part
    time_variation = 2 * 1e10 * np.sin(2 * np.pi * time / 20)**2 # 1/m^3 s
    # Create the mask for t <= 100
    time_mask = (time <= 100).astype(float) # Convert boolean to float (0.0 or 1.0)
    # Tile the altitude dimension first
    q_e_base = np.ones((len(altitude_grid), len(time)))
    # Apply time variation and mask
    q_e_4 = q_e_base * time_variation[np.newaxis, :] * time_mask[np.newaxis, :]
    return q_e_4


output_pdf_filename = 'prog_task_4_v3.pdf'
altitudes_to_run = [110e3, 120e3, 170e3, 230e3]
task_altitudes = {
    0: [110e3, 170e3, 230e3],
    1: [110e3, 170e3, 230e3],
    2: [110e3, 170e3, 230e3],
    3: [110e3, 230e3],
    4: [110e3, 120e3, 170e3, 230e3]
}
altitude_grid = altitude_IRI

with PdfPages(output_pdf_filename) as pdf_final:
    n_init_states = {}
    diagnostic_solutions = {}

    for task in [0, 1, 2, 4]:
        print(f"Running Task {task} (density plots)...")
        if task == 0:
            time = time_0
            n_major_data = n_major(time)
            alpha_vals, k_vals = calculate_rates(Ti, Te, altitude_grid, time)
            q_e_vals = np.tile(1e8, (len(altitude_grid), len(time)))
            q_vals = q_major_i(q_e_vals, n_major_data)
        else:
            time = time_1
            n_major_data = n_major(time)
            alpha_vals, k_vals = calculate_rates(Ti, Te, altitude_grid, time)
            if task == 1:
                q_e_base = np.tile(1e10, (len(altitude_grid), len(time)))
                time_mask = (time <= 100).astype(float)
                q_e_vals = q_e_base * time_mask[np.newaxis, :]
            elif task == 2:
                Ti_adj, Te_adj = Ti.copy(), Te.copy()
                mask = altitude_grid <= 150e3
                Ti_adj[mask] += 1000; Te_adj[mask] += 1000
                Ti_adj[~mask] += 2000; Te_adj[~mask] += 2000
                alpha_vals, k_vals = calculate_rates(Ti_adj, Te_adj, altitude_grid, time)
                q_e_base = np.tile(1e10, (len(altitude_grid), len(time)))
                time_mask = (time <= 100).astype(float)
                q_e_vals = q_e_base * time_mask[np.newaxis, :]
            elif task == 4:
                q_e_vals = q_e_4(time, altitude_grid)
            q_vals = q_major_i(q_e_vals, n_major_data)

        for alt in task_altitudes[task]:
            print(f"  Altitude: {alt/1e3:.0f} km")
            y0 = n_init_states.get(alt, initial_state(alt))
            sol_res = solve_ion_chem(time, n_major_data, q_vals, k_vals, alpha_vals, alt, y0, altitude_grid)
            n_init_states[alt] = sol_res[0].y[:, -1]

            # Density plots (with y-limits 10^1 to 10^12)
            n, _, _, height = sol_res
            fig = plt.figure(figsize=(12,6))
            for i, label in enumerate(['O+', 'O2+', 'N2+', 'NO+', 'NO', 'e-']):
                plt.plot(time, n.y[i], label=label)
            plt.xlabel('Time [s]')
            plt.ylabel(r'Density [m$^{-3}$]')
            plt.title(f'Task {task} - Densities at {height / 1e3:.0f} km')
            plt.legend()
            plt.yscale('log')
            plt.ylim(1e1, 1e12)
            plt.yticks([10 ** i for i in range(1, 13)])
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            pdf_final.savefig(fig)
            plt.close(fig)

            diagnostic_solutions.setdefault(task, []).append((alt, sol_res))

    # Task 3: only decay comparison plots
    print("Running Task 3 (decay analysis plots)...")
    sol_1_110km_res = solve_ion_chem(time_1, n_major(time_1), q_major_i(np.tile(1e10*(time_1<=100), (len(altitude_grid),1)), n_major(time_1)), calculate_rates(Ti, Te, altitude_grid, time_1)[1], calculate_rates(Ti, Te, altitude_grid, time_1)[0], 110e3, n_init_states[110e3], altitude_grid)
    sol_1_230km_res = solve_ion_chem(time_1, n_major(time_1), q_major_i(np.tile(1e10*(time_1<=100), (len(altitude_grid),1)), n_major(time_1)), calculate_rates(Ti, Te, altitude_grid, time_1)[1], calculate_rates(Ti, Te, altitude_grid, time_1)[0], 230e3, n_init_states[230e3], altitude_grid)
    n_major_1 = n_major(time_1)
    alpha_values_1, k_values_1 = calculate_rates(Ti, Te, altitude_grid, time_1)
    time_off_idx = np.argmin(np.abs(time_1 - 100))
    alt_idx_110km = np.argmin(np.abs(altitude_grid - 110e3))
    alt_idx_230km = np.argmin(np.abs(altitude_grid - 230e3))
    n_major_100_110km = [n_major_1[i][alt_idx_110km, time_off_idx] for i in range(3)]
    alpha_100_110km = [alpha_values_1[i][alt_idx_110km, time_off_idx] for i in range(4)]
    k_100_110km = [k_values_1[i][alt_idx_110km, time_off_idx] for i in range(6)]
    n_major_100_230km = [n_major_1[i][alt_idx_230km, time_off_idx] for i in range(3)]
    alpha_100_230km = [alpha_values_1[i][alt_idx_230km, time_off_idx] for i in range(4)]
    k_100_230km = [k_values_1[i][alt_idx_230km, time_off_idx] for i in range(6)]
    n_e_recomb_110, n_e_beta_110, beta_110 = task_3_analysis(sol_1_110km_res, time_1, n_major_100_110km, alpha_100_110km, k_100_110km, 110e3)
    n_e_recomb_230, n_e_beta_230, beta_230 = task_3_analysis(sol_1_230km_res, time_1, n_major_100_230km, alpha_100_230km, k_100_230km, 230e3)
    make_task3_plots(pdf_final, time_1, sol_1_110km_res, sol_1_230km_res,
                     n_e_recomb_110, n_e_beta_110, beta_110,
                     n_e_recomb_230, n_e_beta_230, beta_230)

    # Diagnostic plots (with blank separator page) for tasks 0,1,2,4
    for task in [0,1,2,4]:
        fig_sep = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.5, f'Task {task}', ha='center', va='center', fontsize=20)
        pdf_final.savefig(fig_sep)
        plt.close(fig_sep)
        for alt, sol_res in diagnostic_solutions[task]:
            make_plots(sol_res, pdf_final)

print(f"PDF created: {output_pdf_filename}")
