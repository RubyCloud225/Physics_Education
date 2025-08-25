import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def double_slit(num_points=500, wavelength=1.0, slit_separation=5.0, screen_distance=50.0, measure=False):
    """
    Simulate the double-slit experiment and plot the interference pattern.
    params:
    num_points : int
        Number of points on the screen to calculate intensity.
    wavelength : float
        Wavelength of the particle/wave (λ).
    slit_separation : float
        Distance between the two slits (d).
    screen_distance : float
        Distance from the slits to the screen (L).
    measure : bool
        If True, simulates a measurement collapsing the wavefunction (no interference).

    returns:
    x_positions : ndarray
        Positions on the screen. (in arbitrary units)
    probablities : ndarray
        Normalized probabilities of detecting the particle at each position.
    """

    # Define Screen coordinates (centered around 0)
    x_positions = np.linspace(-20, 20, num_points)

    # Amplitudes from each slit
    # Phase differencce depends on path length difference
    k = 2 * np.pi / wavelength  # wave number
    path_diff = (slit_separation * x_positions) / screen_distance  # path difference
    phase_diff = k * path_diff  # phase difference

    #each slit contributes equally
    amp_slit1 = np.exp(1j * 0)
    amp_slit2 = np.exp(1j * phase_diff)

    if measure:
        prop = np.abs(amp_slit1)**2 + np.abs(amp_slit2)**2
    else:
        total_amp = amp_slit1 + amp_slit2
        prop = np.abs(total_amp)**2

    probabilities = prop / np.max(prop)  # Normalize
    return x_positions, probabilities

def save_results(x, prob, filename_prefix="double_slit"):
    """
    Save results to CSV and plot as PNG.
    """
    # Save CSV
    df = pd.DataFrame({"Position": x, "Probability": prob})
    csv_file = f"{filename_prefix}.csv"
    df.to_csv(csv_file, index=False)

    # Save Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, prob, label="Probability Distribution")
    plt.title("Double-Slit Simulation")
    plt.xlabel("Screen Position")
    plt.ylabel("Normalized Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{filename_prefix}.png")
    plt.close()

    print(f"Results saved to {csv_file} and {filename_prefix}.png")


# -------------------------
# Example Usage
# -------------------------

# Case 1: Without measurement (interference pattern)
x1, prob1 = double_slit(measure=False)
save_results(x1, prob1, filename_prefix="double_slit_interference")

# Case 2: With measurement (no interference)
x2, prob2 = double_slit(measure=True)
save_results(x2, prob2, filename_prefix="double_slit_measured")

"""

This code simulates the double-slit experiment, a fundamental demonstration of quantum mechanics. It calculates the interference pattern formed on a screen when particles (like electrons or photons) pass through two slits. The resulting plot shows the probability distribution of detecting a particle at various positions on the screen, illustrating the wave-like behavior of particles in quantum mechanics.

Key Concepts Illustrated:
1. Wave-Particle Duality: The code demonstrates how particles exhibit wave-like interference patterns.
2. Superposition Principle: The total amplitude at each point on the screen is the sum of amplitudes from both slits.
3. Probability Amplitude: The probability of detecting a particle is related to the square of the amplitude.
4. Phase Difference: The phase difference between the two paths leads to constructive and destructive interference, shaping the probability distribution.
5. Measurement Effect: The code includes an option to simulate a measurement that collapses the wavefunction, resulting in no interference pattern.

To run this code, ensure you have Python installed along with the NumPy and Matplotlib libraries. You can install these libraries using pip if you haven't already:

"""