import numpy as np
from skimage import data, color
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load and preprocess image
image = color.rgb2gray(data.astronaut())
image = resize(image, (128, 128))  # smaller for simplicity
tokens = image.flatten()
tokens /= np.max(tokens)

# Parameters
num_layers = 3
noise_std = 0.005

# Map tokens to qubit amplitudes (|alpha|^2 + |beta|^2 = 1)
alpha = np.sqrt(1 - tokens)        # probability amplitude for |0>
beta = np.sqrt(tokens)             # probability amplitude for |1>
qubits = np.stack([alpha, beta], axis=1)  # shape (num_tokens, 2)

# Store measured tokens at each layer for animation
measured_tokens_layers = []

total_mse = 0.0

# Layered unitary simulation
for layer in range(num_layers):
    # Random single-qubit rotations per token
    theta = np.random.normal(0, 0.05, size=qubits.shape[0])  # small rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    alpha_new = cos_theta * qubits[:,0] - sin_theta * qubits[:,1]
    beta_new  = sin_theta * qubits[:,0] + cos_theta * qubits[:,1]
    
    # Add small stochastic noise
    alpha_new += np.random.normal(0, noise_std, size=alpha_new.shape)
    beta_new  += np.random.normal(0, noise_std, size=beta_new.shape)
    
    # Normalize qubit
    norm = np.sqrt(alpha_new**2 + beta_new**2)
    qubits[:,0] = alpha_new / norm
    qubits[:,1] = beta_new / norm

    # Measurement (simulate probabilistic collapse) at this layer
    measured_tokens = qubits[:,1]**2  # probability of |1> = token value
    measured_tokens_layers.append(measured_tokens.reshape(image.shape))
    
    # Save measured tokens of this layer as .npy file
    np.save(f'measured_tokens_layer{layer+1}.npy', measured_tokens.reshape(image.shape))
    
    # Calculate MSE loss between measured tokens and original tokens
    mse = np.mean((measured_tokens - tokens)**2)
    total_mse += mse
    
    # Calculate fidelity as squared overlap between original and measured qubit states
    # Original qubit states: [sqrt(1 - tokens), sqrt(tokens)]
    original_alpha = np.sqrt(1 - tokens)
    original_beta = np.sqrt(tokens)
    overlap = original_alpha * qubits[:,0] + original_beta * qubits[:,1]
    fidelity = np.mean(np.abs(overlap)**2)
    
    print(f'Layer {layer+1}: MSE = {mse:.6f}, Fidelity = {fidelity:.6f}')

print(f'Total MSE after {num_layers} layers: {total_mse:.6f}')

# Visualization of final reconstructed image
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(measured_tokens_layers[-1], cmap='gray')
plt.title('Qubit Reconstructed Image')
plt.axis('off')
plt.tight_layout()

# Save the final reconstructed image as PNG
plt.savefig('qubit_reconstruction.png')

plt.show()

# Animation of layer-by-layer reconstruction
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(measured_tokens_layers[0], cmap='gray', vmin=0, vmax=1)
ax.set_title('Layer 1 Reconstruction')
ax.axis('off')

def update(frame):
    im.set_data(measured_tokens_layers[frame])
    ax.set_title(f'Layer {frame+1} Reconstruction')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=num_layers, blit=True, repeat=False)

# Save the animation as mp4 using ffmpeg writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
ani.save('qubit_reconstruction.mp4', writer=writer)

plt.close(fig)