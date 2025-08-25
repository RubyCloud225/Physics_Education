import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Example Data Tokens
# -------------------------
tokens = np.array([0.1, 0.5, 0.9, 0.3])  # Normalized token values

# Simulation parameters
wave_amplitude = 0.05
num_layers = 3
noise_std = 0.01  # Standard deviation of Gaussian noise

# ------------------------
# Amplitudes
#-------------------------

amplitudes = np.ones_like(tokens)

hybrid_state = tokens + wave_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, size=tokens.shape))

# -------------------------
# Gaussian "Lossless" Compression Function
# -------------------------
def compress_gaussian(state, noise_std):
    """
    Add Gaussian noise to simulate stochastic encoding,
    without dropping any components (lossless).
    """
    noise = np.random.normal(0, noise_std, size=state.shape)
    return state + noise

# -------------------------
# Layered Processing with Waves + Unitary + Gaussian Noise
# -------------------------
hybrid_state = tokens + wave_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, size=tokens.shape))

layer_losses = []
layer_measurements = []
layer_amplitudes = [amplitudes.copy()]

for layer in range(num_layers):
    # Generate a random unitary-like matrix
    z = np.random.randn(len(tokens), len(tokens)) + 1j * np.random.randn(len(tokens), len(tokens))
    U, _ = np.linalg.qr(z)
    
    # Apply unitary transformation with soft mix
    hybrid_state = 0.5 * amplitudes * (U @ hybrid_state) + 0.5 * hybrid_state
    
    # Apply Gaussian noise (lossless stochastic compression)
    hybrid_state = compress_gaussian(hybrid_state, noise_std)
    
    # Measurement for this layer
    measured_tokens = np.abs(np.real(hybrid_state))
    measured_tokens /= np.max(measured_tokens)
    measured_tokens *= np.max(tokens)
    
    # Compute MSE loss for this layer
    loss = np.mean((tokens - measured_tokens)**2)

    amplitudes = amplitudes * (1 + 0.1 * (measured_tokens - amplitudes)) # Multiplicative soft update
    
    amplitudes = np.clip(amplitudes, 0, 1)  # Keep amplitudes normalized
    
    layer_losses.append(loss)
    layer_measurements.append(np.round(measured_tokens, 3))
    layer_amplitudes.append(amplitudes.copy())
    losss = layer_losses[-1]
    
    print(f"Layer {layer+1} Measurement: {layer_measurements[-1]}, Loss: {loss:.4f}, amplitudes: {np.round(amplitudes,3)}")

# -------------------------
# Final Output
# -------------------------
final_tokens = layer_measurements[-1]
final_loss = layer_losses[-1]
print("\nFinal Tokens after all layers: ", final_tokens)
print("Final MSE Loss: ", final_loss)

# -------------------------
# Visualization of per-layer data
# -------------------------
layers = np.arange(1, num_layers+1)
tokens_per_layer = np.array(layer_measurements)
amplitudes_per_layer = np.array(layer_amplitudes[1:])  # skip initial amplitudes
losses_per_layer = np.array(layer_losses)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot per-layer measured tokens
for i in range(tokens.shape[0]):
    axs[0].plot(layers, tokens_per_layer[:, i], marker='o', label=f'Token {i+1}')
axs[0].set_title('Per-layer Measured Tokens')
axs[0].set_xlabel('Layer')
axs[0].set_ylabel('Measured Token Value')
axs[0].legend()
axs[0].grid(True)

# Plot per-layer amplitudes
for i in range(amplitudes.shape[0]):
    axs[1].plot(np.arange(0, num_layers+1), np.array(layer_amplitudes)[:, i], marker='o', label=f'Amplitude {i+1}')
axs[1].set_title('Per-layer Amplitudes')
axs[1].set_xlabel('Layer')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True)

# Plot per-layer MSE loss
axs[2].plot(layers, losses_per_layer, marker='o', color='red', label='MSE Loss')
axs[2].set_title('Per-layer MSE Loss')
axs[2].set_xlabel('Layer')
axs[2].set_ylabel('MSE Loss')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('wavemeasurement_layers.png')
plt.show()