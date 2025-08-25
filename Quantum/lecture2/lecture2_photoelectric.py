import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
h_eV = 4.135667e-15  # Planck's constant in eV.s
h_J = 6.62607015e-34  # Planck's constant in J.s
work_function_eV = 2.0  # Work function in eV (example value)
work_function_J = 1.60218e-19  # Work function in eV (example value)

def photoelectric_effect(frequencies, intensity=1.0, measure=False, unit='eV'):
    """
    Simulate the photoelectric effect and plot the kinetic energy of emitted electrons.
    
    params:
    frequencies : ndarray
        Array of light frequencies (in Hz).
    intensity : float
        Intensity of the incident light (affects electron *count*, not kinetic energy).
    measure : bool
        If True, simulates a measurement collapsing the wavefunction (no kinetic energy spread).
    unit : str
        Unit for Planck's constant ('eV' or 'J').
    
    returns:
    Kinetic_energies : ndarray
        Kinetic energies of emitted electrons (in specified Unit).
    electron_counts : ndarray
        Number of emitted electrons (arbitrary units). 
    """

    if unit == 'J':
        h = h_J
        work_function = work_function_J
    else:
        h = h_eV
        work_function = work_function_eV
    # Calculate photon energies
    photon_energies = h * frequencies  # in specified unit

    # Calculate kinetic energies of emitted electrons
    kinetic_energies = np.maximum(0, photon_energies - work_function)

    # Simulate electron counts based on intensity and whether a measurement is made
    if measure:
        electron_counts = intensity * (photon_energies > work_function).astype(float)
    else:
        electron_counts = intensity * (photon_energies > work_function).astype(float) * np.random.poisson(5, size=frequencies.shape)

    return kinetic_energies, electron_counts

def save_results(frequencies, kinetic_energies, electron_counts, filename_prefix="photoelectric_effect", unit='eV'):
    """
    Save results to CSV and plot as PNG.
    """
    energy_label = "Kinetic Energy (eV)" if unit == 'eV' else "Kinetic Energy (J)"
    # Save CSV
    df = pd.DataFrame({
        "Frequency (Hz)": frequencies,
        energy_label: kinetic_energies,
        "Electron Count": electron_counts
    })
    csv_file = f"{filename_prefix}.csv"
    df.to_csv(csv_file, index=False)

    # Save Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(frequencies, kinetic_energies, c='blue', label="Kinetic Energy of Electrons")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(energy_label)
    plt.title("Photoelectric Effect: Kinetic Energy vs Frequency")
    plt.axhline(0, color='red', linestyle='--', label="Work Function Threshold")
    plt.legend()
    plt.grid()
    plt.savefig(f"{filename_prefix}.png")
    plt.close()

# Example usage
if __name__ == "__main__":
    # Frequencies from 0 to 1e15 Hz
    frequencies = np.linspace(0, 1e15, 500)
    
    # Simulate without measurement
    # Simulate without measurement in eV
    kinetic_energies, electron_counts = photoelectric_effect(frequencies, intensity=10.0, measure=False, unit='eV')
    save_results(frequencies, kinetic_energies, electron_counts, filename_prefix="photoelectric_no_measure", unit='eV')

    # Simulate with measurement
    # Simulate with measurement in eV
    kinetic_energies_meas, electron_counts_meas = photoelectric_effect(frequencies, intensity=10.0, measure=True, unit='eV')
    save_results(frequencies, kinetic_energies_meas, electron_counts_meas, filename_prefix="photoelectric_with_measure", unit='eV')
        # Simulate without measurement in Joules
    kinetic_energies_J, electron_counts_J = photoelectric_effect(frequencies, intensity=10.0, measure=False, unit='J')
    save_results(frequencies, kinetic_energies_J, electron_counts_J, filename_prefix="photoelectric_no_measure_J", unit='J')

    # Simulate with measurement in Joules
    kinetic_energies_meas_J, electron_counts_meas_J = photoelectric_effect(frequencies, intensity=10.0, measure=True, unit='J')
    save_results(frequencies, kinetic_energies_meas_J, electron_counts_meas_J, filename_prefix="photoelectric_with_measure_J", unit='J')
