import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.animation as animation

# -------------------------
# Load and preprocess image
# -------------------------
image = rgb2gray(data.astronaut())  # example image
image = resize(image, (128, 128))     # downscale to 128x128 for simplicity

# -------------------------
# Wave-dimension lossless compression
# -------------------------
def wave_dimension_compress(img, wave_dim=4):
    # Compress image by averaging over wave_dim blocks along rows
    h, w = img.shape
    assert h % wave_dim == 0, "Image height must be divisible by wave_dim"
    compressed = np.mean(img.reshape(h // wave_dim, wave_dim, w), axis=1)
    return compressed

def wave_dimension_decompress(compressed, wave_dim=4):
    # Decompress by resizing with bilinear interpolation to smooth the image
    decompressed = resize(
        compressed,
        (compressed.shape[0] * wave_dim, compressed.shape[1]),
        order=1, mode='reflect', anti_aliasing=True, preserve_range=True
    )
    return decompressed

# Use higher wave_dim for better pixel resolution
wave_dim = 4
compressed = wave_dimension_compress(image, wave_dim=wave_dim)

# Simulation parameters
wave_amplitude = 0.05
num_layers = 3
# Slightly reduced noise for stability
noise_std = 0.005
# Amplitude initialization
amplitudes = np.ones_like(compressed)

# Gaussian "lossless" noise function
def compress_gaussian(state, noise_std):
    noise = np.random.normal(0, noise_std, size=state.shape)
    return state + noise

# Initialize hybrid state
hybrid_state = compressed + wave_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, size=compressed.shape))

# Layered processing
layer_measurements = []
layer_losses = []
layer_amplitudes = [amplitudes.copy()]

per_layer_mse = []
for layer in range(num_layers):
    z = np.random.randn(*compressed.shape) + 1j * np.random.randn(*compressed.shape)
    U, _ = np.linalg.qr(z)

    # Store pre-layer compressed for loss calculation
    pre_layer_compressed = compressed.copy()

    # Apply unitary with soft mix
    hybrid_state = 0.5 * amplitudes * (U @ hybrid_state) + 0.5 * hybrid_state

    # Add Gaussian noise
    hybrid_state = compress_gaussian(hybrid_state, noise_std)

    # Measurement
    measured_tokens = np.abs(np.real(hybrid_state))
    measured_tokens *= np.max(compressed)

    # Compute MSE loss between pre-layer compressed and measured tokens
    loss = np.mean((pre_layer_compressed - measured_tokens) ** 2)

    # Amplitude update, with minimum amplitude floor to prevent collapse
    amplitudes = amplitudes * (1 + 0.1 * (measured_tokens - amplitudes))
    amplitudes = np.clip(amplitudes, 0.1, 1.0)

    # Update compressed for next layer
    compressed = measured_tokens.copy()

    layer_measurements.append(measured_tokens.copy())
    layer_losses.append(loss)
    layer_amplitudes.append(amplitudes.copy())

    # --- Per-layer MSE: reconstruct image from measured tokens ---
    recon_img = wave_dimension_decompress(measured_tokens, wave_dim=wave_dim)
    recon_img = np.clip(recon_img, 0, 1)
    mse_img = np.mean((image - recon_img) ** 2)
    per_layer_mse.append(mse_img)
    print(f"Layer {layer+1} Loss: {loss:.5f}   Reconstructed MSE: {mse_img:.5f}")

# -------------------------
# Reconstruct final image via wave-dimension inverse transform
# -------------------------
# Reconstruct from final measured tokens
final_tokens_compressed = layer_measurements[-1]
reconstructed_image = wave_dimension_decompress(final_tokens_compressed, wave_dim=wave_dim)
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# -------------------------
# Visualize original vs reconstructed
# -------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(reconstructed_image, cmap='gray')
axs[1].set_title('Reconstructed Image')
axs[1].axis('off')
plt.tight_layout()
plt.savefig('image_reconstruction.png')

# Print per-layer MSEs for reference
print("Per-layer reconstructed image MSEs:")
for i, mse in enumerate(per_layer_mse):
    print(f"  Layer {i+1}: MSE = {mse:.5f}")

# -------------------------
# Animation of layer-by-layer reconstruction
# -------------------------
fig_anim, ax_anim = plt.subplots(figsize=(5,5))
ax_anim.axis('off')
im = ax_anim.imshow(np.zeros_like(image), cmap='gray', vmin=0, vmax=1)

def update(frame):
    tokens = layer_measurements[frame]
    img = wave_dimension_decompress(tokens, wave_dim=wave_dim)
    img = np.clip(img, 0, 1)
    im.set_data(img)
    ax_anim.set_title(f'Layer {frame+1} Reconstruction')
    return [im]

ani = animation.FuncAnimation(fig_anim, update, frames=num_layers, blit=True, repeat=False)
ani.save('image_reconstruction_animation.mp4', writer='ffmpeg', fps=1)

plt.show()