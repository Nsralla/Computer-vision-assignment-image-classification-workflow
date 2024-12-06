import matplotlib.pyplot as plt

# Updated data
kernel_sizes = [3, 5, 7]
mse = [85.9307, 91.6422, 95.2886]
psnr = [28.7893, 28.5099, 28.3404]
edge_density_filtered = [0.3052, 0.1041, 0.0286]
ssim_noisy = [0.1175, 0.1175, 0.1175]  # Constant for this data
ssim_filtered = [0.1219, 0.1391, 0.1336]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot MSE vs Kernel Size
axes[0, 0].plot(kernel_sizes, mse, marker='o', color='blue', linestyle='-')
axes[0, 0].set_title('MSE vs Kernel Size')
axes[0, 0].set_xlabel('Kernel Size')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].grid(True)

# Plot PSNR vs Kernel Size
axes[0, 1].plot(kernel_sizes, psnr, marker='s', color='green', linestyle='--')
axes[0, 1].set_title('PSNR vs Kernel Size')
axes[0, 1].set_xlabel('Kernel Size')
axes[0, 1].set_ylabel('PSNR (dB)')
axes[0, 1].grid(True)

# Plot Edge Density vs Kernel Size
axes[1, 0].plot(kernel_sizes, edge_density_filtered, marker='^', color='red', linestyle='-.')
axes[1, 0].set_title('Edge Density (Filtered) vs Kernel Size')
axes[1, 0].set_xlabel('Kernel Size')
axes[1, 0].set_ylabel('Edge Density')
axes[1, 0].grid(True)

# Plot SSIM (Original vs Filtered) vs Kernel Size
axes[1, 1].plot(kernel_sizes, ssim_filtered, marker='d', color='purple', linestyle=':')
axes[1, 1].set_title('SSIM (Original vs Filtered) vs Kernel Size')
axes[1, 1].set_xlabel('Kernel Size')
axes[1, 1].set_ylabel('SSIM')
axes[1, 1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()
