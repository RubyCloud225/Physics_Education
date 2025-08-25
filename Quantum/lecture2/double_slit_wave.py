import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------
# Constants
# -------------------------
h = 4.135667e-15  # Planck's constant in eV·s
work_function = 2.0  # Example metal in eV

# Simulation parameters
num_electrons = 5000     # Total electrons to simulate
screen_points = 500      # Positions on screen
slit_separation = 5.0    # Arbitrary units
screen_distance = 50.0   # Distance to screen
light_frequency = 6e14   # Hz, above threshold for emission

# -------------------------
# Step 1: Check if electrons are emitted (Photoelectric effect)
# -------------------------
photon_energy = h * light_frequency
if photon_energy < work_function:
    raise ValueError("Photon energy below work function: no electrons emitted!")
else:
    print(f"Photon energy {photon_energy:.2f} eV > Work function {work_function} eV → electrons emitted.")

# -------------------------
# Step 2: Build probability distribution for double-slit interference
# -------------------------
x_positions = np.linspace(-20, 20, screen_points)
k = 2 * np.pi / (screen_distance / slit_separation)  # simple wavelength approximation

# Complex amplitudes from two slits
amp_slit1 = np.exp(1j * 0)
amp_slit2 = np.exp(1j * (k * x_positions))

# Total probability (normalized)
prob_dist = np.abs(amp_slit1 + amp_slit2)**2
prob_dist /= np.sum(prob_dist)

# -------------------------
# Step 3: Simulate electron hits
# -------------------------
electron_hits = np.random.choice(x_positions, size=num_electrons, p=prob_dist)

# -------------------------
# Step 4: Plot the accumulated hits (wave-like pattern)
# -------------------------
plt.figure(figsize=(10,5))
plt.hist(electron_hits, bins=screen_points, color='blue', alpha=0.7)
plt.title("Electron Build-Up: Photoelectric + Double-Slit Interference")
plt.xlabel("Screen Position (arbitrary units)")
plt.ylabel("Electron Counts")
plt.grid(True, alpha=0.3)
plt.show()