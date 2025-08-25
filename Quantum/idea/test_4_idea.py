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

# Function to process a patch of the image with stabilized Schrödinger-inspired local patch evolution
def process_patch(patch_image):
    patch_height, patch_width = patch_image.shape
    patch_image = np.clip(patch_image, 0.0, 1.0)
    amplitude = np.sqrt(patch_image)  # amplitude proportional to sqrt(pixel intensity)
    phase = np.linspace(0, np.pi/2, patch_image.size).reshape(patch_image.shape)  # simple phase gradient

    A = amplitude.flatten()
    phi = phase.flatten()
    alpha = np.cos(A)
    beta = np.exp(1j*phi) * np.sin(A)
    qubits = np.stack([alpha, beta], axis=1)  # shape (num_qubits, 2)
    num_qubits = qubits.shape[0]

    qubit_alpha = qubits[:,0].astype(complex).copy()
    qubit_beta = qubits[:,1].astype(complex).copy()

    # Parameterized rotations with small random angles for stability
    angles = np.random.normal(0, 0.03, size=(num_qubits, 3))
    qubit_amplitudes = np.stack([qubit_alpha, qubit_beta], axis=1)
    qubit_amplitudes = single_qubit_rotation(qubit_amplitudes, angles)
    qubit_alpha = qubit_amplitudes[:,0]
    qubit_beta = qubit_amplitudes[:,1]

    # Removed normalization after rotations

    # Apply local unitary operator encoding neighbor interactions
    qubits = np.stack([qubit_alpha, qubit_beta], axis=1)
    qubits = local_unitary_operator(qubits, patch_height, patch_width)
    qubit_alpha = qubits[:,0]
    qubit_beta = qubits[:,1]

    # Removed normalization after local unitary operator

    # Weighted diffusion step with small diffusion weight
    diffusion_weight = 0.1
    qubit_alpha, qubit_beta = weighted_diffusion(qubit_alpha, qubit_beta, patch_height, patch_width, diffusion_weight)

    # Removed normalization after diffusion

    # Add small fixed Gaussian noise for stability
    qubit_alpha += np.random.normal(0, noise_std, size=qubit_alpha.shape)
    qubit_beta += np.random.normal(0, noise_std, size=qubit_beta.shape)

    # Removed normalization after noise

    # Residual error capture and conservative correction only where magnitude is above threshold
    residual = patch_image.flatten() - np.abs(qubit_beta)**2
    mask = np.sqrt(np.abs(qubit_alpha)**2 + np.abs(qubit_beta)**2) > 1e-6
    qubit_beta[mask] += residual[mask] * correction_factor

    # Removed normalization after residual correction

    # Measurement: probability of |1> state as reconstructed pixel intensity
    measured_tokens = np.abs(qubit_beta)**2

    # Reconstruct patch image from measured tokens
    reconstructed_patch = measured_tokens.reshape(patch_image.shape)

    # Calculate MSE and fidelity
    mse = np.mean((measured_tokens - patch_image.flatten())**2)

    original_alpha = np.cos(amplitude.flatten())
    original_beta = np.exp(1j*phase.flatten()) * np.sin(amplitude.flatten())
    overlap = original_alpha * qubit_alpha + np.conj(original_beta) * qubit_beta
    fidelity = np.mean(np.abs(overlap)**2)

    return reconstructed_patch, mse, fidelity, residual.reshape(patch_image.shape)

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

            reconstructed_patch, mse_patch, fidelity_patch, residual_patch = process_patch(patch_input)

            reconstructed_image[row_start:row_end, col_start:col_end] = reconstructed_patch
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
wave_amplitudes = np.arctan2(np.sqrt(reconstructed_image), np.sqrt(1 - reconstructed_image))
wave_trend = gaussian_filter(wave_amplitudes + residual_full * decoder_gain, sigma=1.0)
reconstructed_image_decoded = np.sin(wave_trend)**2

# Visualization of final reconstructed image (after decoder)
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

plt.savefig('vqc_reconstruction.png')
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
ani.save('vqc_reconstruction.mp4', writer=writer)

plt.close(fig)