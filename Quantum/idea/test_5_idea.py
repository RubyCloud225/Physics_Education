import numpy as np
from skimage import data, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Load and preprocess image
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256))  # smaller for VQC

patch_size = 8
height, width = image.shape
num_layers = 2

correction_factor = 0.02  # Reduced for conservative residual correction
decoder_gain = 0.1        # For decoder residual gain
noise_std = 0.005         # Small fixed Gaussian noise for stability

# Parameterized single-qubit rotations: Rx, Ry, Rz with small random angles
def single_qubit_rotation(qubit_amplitudes, angles):
    # angles: array of shape (num_qubits, 3) for Rx, Ry, Rz
    Rx_angles = angles[:, 0]
    Ry_angles = angles[:, 1]
    Rz_angles = angles[:, 2]

    # Define rotation matrices for single qubit
    def Rx(theta):
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]])

    def Ry(theta):
        return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]])

    def Rz(theta):
        return np.array([[np.exp(-1j*theta/2), 0],
                         [0, np.exp(1j*theta/2)]])

    rotated_qubits = np.zeros_like(qubit_amplitudes, dtype=complex)
    for i in range(qubit_amplitudes.shape[0]):
        q = qubit_amplitudes[i]
        U = Rz(Rz_angles[i]) @ Ry(Ry_angles[i]) @ Rx(Rx_angles[i])
        rotated_qubits[i] = U @ q
    return rotated_qubits

# Local unitary operator per patch encoding neighbor interactions
def local_unitary_operator(qubits, patch_height, patch_width):
    num_qubits = qubits.shape[0]
    qubit_alpha = qubits[:,0].copy()
    qubit_beta = qubits[:,1].copy()

    # Apply small-angle partial entangling rotations within patch (2x2 neighborhoods)
    theta_ent = 0.05  # smaller angle for stability
    for row in range(patch_height - 1):
        for col in range(patch_width - 1):
            idx1 = row*patch_width + col
            idx2 = row*patch_width + col + 1
            idx3 = (row+1)*patch_width + col
            idx4 = (row+1)*patch_width + col + 1

            # Entangle horizontally in top row
            q1 = np.array([qubit_alpha[idx1], qubit_beta[idx1]])
            q2 = np.array([qubit_alpha[idx2], qubit_beta[idx2]])
            q1_new, q2_new = partial_entangling_rotation(q1, q2, theta_ent)
            qubit_alpha[idx1], qubit_beta[idx1] = q1_new
            qubit_alpha[idx2], qubit_beta[idx2] = q2_new

            # Entangle vertically in left column
            q3 = np.array([qubit_alpha[idx1], qubit_beta[idx1]])
            q4 = np.array([qubit_alpha[idx3], qubit_beta[idx3]])
            q3_new, q4_new = partial_entangling_rotation(q3, q4, theta_ent)
            qubit_alpha[idx1], qubit_beta[idx1] = q3_new
            qubit_alpha[idx3], qubit_beta[idx3] = q4_new

            # Entangle diagonally
            q5 = np.array([qubit_alpha[idx1], qubit_beta[idx1]])
            q6 = np.array([qubit_alpha[idx4], qubit_beta[idx4]])
            q5_new, q6_new = partial_entangling_rotation(q5, q6, theta_ent)
            qubit_alpha[idx1], qubit_beta[idx1] = q5_new
            qubit_alpha[idx4], qubit_beta[idx4] = q6_new

    # Normalize after unitary
    #norm = np.sqrt(np.abs(qubit_alpha)**2 + np.abs(qubit_beta)**2)
    #norm = np.maximum(norm, 1e-12)
    #qubit_alpha /= norm
    #qubit_beta /= norm

    return np.stack([qubit_alpha, qubit_beta], axis=1)

# Partial entangling rotation on two qubits within 2x2 patches
def partial_entangling_rotation(q1, q2, theta):
    # q1, q2: arrays shape (2,), complex amplitudes
    # Apply a controlled-phase-like partial entangling rotation
    # Construct a simple entangling unitary on two qubits' amplitudes
    # Here, we approximate with a partial swap controlled by theta

    # Construct 4D state vector from two qubits
    # |00>, |01>, |10>, |11> amplitudes
    state = np.zeros(4, dtype=complex)
    state[0] = q1[0]*q2[0]
    state[1] = q1[0]*q2[1]
    state[2] = q1[1]*q2[0]
    state[3] = q1[1]*q2[1]

    # Partial swap between |01> and |10> states
    c, s = np.cos(theta), np.sin(theta)
    new_state = state.copy()
    new_state[1] = c*state[1] - 1j*s*state[2]
    new_state[2] = -1j*s*state[1] + c*state[2]

    # Decompose back to qubit amplitudes (approximate by partial trace)
    # For simplicity, we extract reduced states by summing over partner qubit
    q1_new = np.array([new_state[0] + new_state[1], new_state[2] + new_state[3]])
    q2_new = np.array([new_state[0] + new_state[2], new_state[1] + new_state[3]])

    # Normalize individual qubits
    q1_new /= np.linalg.norm(q1_new) if np.linalg.norm(q1_new) > 0 else 1
    q2_new /= np.linalg.norm(q2_new) if np.linalg.norm(q2_new) > 0 else 1

    return q1_new, q2_new

# Weighted diffusion using averaging for neighbors
def weighted_diffusion(qubit_alpha, qubit_beta, height, width, diffusion_weight):
    alpha_new = np.zeros_like(qubit_alpha, dtype=complex)
    beta_new = np.zeros_like(qubit_beta, dtype=complex)
    for idx in range(len(qubit_alpha)):
        row = idx // width
        col = idx % width
        neighbors = []
        if row > 0:
            neighbors.append((row-1)*width + col)
        if row < height - 1:
            neighbors.append((row+1)*width + col)
        if col > 0:
            neighbors.append(row*width + (col-1))
        if col < width -1:
            neighbors.append(row*width + (col+1))

        # Include self in averaging
        neighbors.append(idx)
        total_neighbors = len(neighbors)

        sum_alpha = np.sum(qubit_alpha[neighbors])
        sum_beta = np.sum(qubit_beta[neighbors])

        alpha_new[idx] = (1 - diffusion_weight) * qubit_alpha[idx] + diffusion_weight * sum_alpha / total_neighbors
        beta_new[idx] = (1 - diffusion_weight) * qubit_beta[idx] + diffusion_weight * sum_beta / total_neighbors

    # Normalize after diffusion
    #norm = np.sqrt(np.abs(alpha_new)**2 + np.abs(beta_new)**2)
    #alpha_new /= norm
    ##norm = np.maximum(norm, 1e-12)
    #beta_new /= norm

    return alpha_new, beta_new

def process_patch(patch_image, dt=0.01, num_steps=5, gain=0.02):
    patch_height, patch_width = patch_image.shape
    patch_image = np.clip(patch_image, 0.0, 1.0)

    # Initialize wavefunction with amplitude sqrt of pixel intensity and random phase
    amplitude = np.sqrt(patch_image)
    phase = np.random.uniform(0, 2*np.pi, size=patch_image.shape)
    psi = amplitude * np.exp(1j*phase)
    
    # Construct discrete Laplacian as simple Hamiltonian for patch
    H = np.zeros((patch_height*patch_width, patch_height*patch_width), dtype=complex)
    for row in range(patch_height):
        for col in range(patch_width):
            idx = row*patch_width + col
            H[idx, idx] = -4
            if row > 0: H[idx, idx - patch_width] = 1
            if row < patch_height-1: H[idx, idx + patch_width] = 1
            if col > 0: H[idx, idx - 1] = 1
            if col < patch_width-1: H[idx, idx + 1] = 1

    psi_flat = psi.flatten()
    residual_flat = patch_image.flatten() - np.abs(psi_flat)**2

    # Dynamic lambda-based evolution
    evolve = lambda psi, H, dt, residual, gain: psi + dt * (-1j * H @ psi) + gain * residual

    for _ in range(num_steps):
        # Adaptive dt based on local residual magnitude
        dt_step = dt * (1 + 5 * np.abs(residual_flat))
        psi_flat = evolve(psi_flat, H, dt_step, residual_flat, gain)
        residual_flat = patch_image.flatten() - np.abs(psi_flat)**2

    psi = psi_flat.reshape(patch_height, patch_width)
    reconstructed_patch = np.abs(psi)**2

    return reconstructed_patch, np.mean((reconstructed_patch - patch_image)**2), residual_flat.reshape(patch_height, patch_width)

def compress_patch(patch, compression_factor=2):
    """Downscale and upscale patch to simulate compression."""
    h, w = patch.shape
    h_small, w_small = h // compression_factor, w // compression_factor
    patch_small = resize(patch, (h_small, w_small), anti_aliasing=True)
    patch_compressed = resize(patch_small, (h, w), anti_aliasing=True)
    return patch_compressed

measured_tokens_layers = []
total_mse = 0.0
total_fidelity = 0.0
residual_full = np.zeros_like(image)

for layer in range(num_layers):

    reconstructed_image = np.zeros_like(image)
    residual_accum = np.zeros_like(image)
    mse_accum = 0.0
    fidelity_accum = 0.0
    count_patches = 0

    for row_start in range(0, height, patch_size):
        for col_start in range(0, width, patch_size):
            row_end = min(row_start + patch_size, height)
            col_end = min(col_start + patch_size, width)

            patch = image[row_start:row_end, col_start:col_end]

            # Add residual from previous layer to patch input for correction
            patch_input = patch + residual_full[row_start:row_end, col_start:col_end]

            reconstructed_patch, mse_patch, residual_patch = process_patch(patch_input)
            fidelity_patch = 1.0  # Placeholder since fidelity is not computed in wave approach

            # Apply compression to reconstructed patch
            reconstructed_patch_compressed = compress_patch(reconstructed_patch, compression_factor=2)

            reconstructed_image[row_start:row_end, col_start:col_end] = reconstructed_patch_compressed
            residual_accum[row_start:row_end, col_start:col_end] = residual_patch

            mse_accum += mse_patch
            fidelity_accum += fidelity_patch
            count_patches += 1

    avg_mse = mse_accum / count_patches
    avg_fidelity = fidelity_accum / count_patches
    total_mse += avg_mse
    total_fidelity = avg_fidelity  # last layer fidelity

    residual_full = image - reconstructed_image

    measured_tokens_layers.append(reconstructed_image)

    print(f'Layer {layer+1}: MSE = {avg_mse:.6f}, Fidelity = {avg_fidelity:.6f}')

print(f'Total MSE after {num_layers} layers: {total_mse:.6f}')

# Step 5: Decoder - convert final qubit states back into wave amplitudes and use wave trend for reconstruction
# Clip reconstructed image to [0,1] to prevent invalid sqrt
reconstructed_image_clipped = np.clip(reconstructed_image, 0.0, 1.0)
#wave_amplitudes = np.arctan2(
#    np.sqrt(reconstructed_image_clipped),
#    np.sqrt(1.0 - reconstructed_image_clipped)
#)
#wave_trend = gaussian_filter(wave_amplitudes + residual_full * decoder_gain, sigma=1.0)
#reconstructed_image_decoded = np.clip(np.sin(wave_trend)**2, 0.0, 1.0)
psi = np.sqrt(reconstructed_image_clipped)  # initial amplitude for wavefunction
# Visualization of final reconstructed image (after decoder)
# Construct discrete Laplacian as Hamiltonian
h, w = psi.shape
H = np.zeros((h*w, h*w), dtype=complex)
for row in range(h):
    for col in range(w):
        idx = row*w + col
        H[idx, idx] = -4
        if row > 0: H[idx, idx - w] = 1
        if row < h-1: H[idx, idx + w] = 1
        if col > 0: H[idx, idx - 1] = 1
        if col < w-1: H[idx, idx + 1] = 1

psi_flat = psi.flatten()
dt = 0.02
psi_flat = psi_flat + (-1j) * dt * (H @ psi_flat)  # simple Hamiltonian pulse
psi = psi_flat.reshape(h, w)
reconstructed_image_decoded = np.clip(np.abs(psi)**2, 0.0, 1.0)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(reconstructed_image_decoded, cmap='gray')
plt.title('VQC Reconstructed Image (Decoded)')
plt.axis('off')
plt.tight_layout()

plt.savefig('vqc_reconstruction1.png')
plt.show()

# Step 6: Animation of layer-by-layer reconstruction
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(measured_tokens_layers[0], cmap='gray', vmin=0, vmax=1)
ax.set_title('Layer 1 Reconstruction')
ax.axis('off')

def update(frame):
    im.set_data(measured_tokens_layers[frame])
    ax.set_title(f'Layer {frame+1} Reconstruction')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=num_layers, blit=True, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
ani.save('vqc_reconstruction1.mp4', writer=writer)

plt.close(fig)