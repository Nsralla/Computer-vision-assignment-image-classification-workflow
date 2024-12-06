import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from config import noisy_image_path, original_image_path, noise_name
def apply_median_filter(image, kernel_size):
    start_time = time.time()  # Start timing
    filtered_image = cv2.medianBlur(image, kernel_size)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    return filtered_image, elapsed_time

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')  # Infinite PSNR when there is no error (identical images)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def canny_edge_detection(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def plot_images():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title(f'{noise_name} Noisy Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Median Filtered Image (Kernel Size = {kernel_size})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_canny():
    plt.figure(figsize=(18, 8))  

    # Original Image and its edges
    plt.subplot(2, 3, 1)  # First row, first column
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 3, 4)  # Second row, first column
    plt.imshow(edges_original, cmap='gray')
    plt.title('Edges of Original Image')
    plt.axis('off')

    # Noisy Image and its edges
    plt.subplot(2, 3, 2)  # First row, second column
    plt.imshow(noisy_image, cmap='gray')
    plt.title(f'{noise_name} Noisy Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 3, 5)  # Second row, second column
    plt.imshow(edges_noisy, cmap='gray')
    plt.title('Edges of Noisy Image')
    plt.axis('off')

    # Filtered Image and its edges
    plt.subplot(2, 3, 3)  # First row, third column
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'paper_salt_medium_noise filtered Image (Kernel Size = {kernel_size})')
    plt.axis('off')

    plt.subplot(2, 3, 6)  # Second row, third column
    plt.imshow(edges_filtered, cmap='gray')
    plt.title('Edges of Filtered Image')
    plt.axis('off')

    plt.tight_layout()  
    plt.show()

# Load the Noisy Image
noisy_image_path = noisy_image_path
noisy_image = cv2.imread(noisy_image_path, cv2.IMREAD_GRAYSCALE)

# Load the Original Clean Image
original_image_path = original_image_path
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

kernel_size = 7 # could be 3, 5, 7
filtered_image, filter_time = apply_median_filter(noisy_image, kernel_size)

# Compute MSE and PSNR
mse = calculate_mse(original_image, filtered_image)
psnr = calculate_psnr(mse)

print("MSE:", mse)
print("PSNR:", psnr)
print("Filtering time: {:.4f} seconds".format(filter_time))

# Edge Detection
edges_original = canny_edge_detection(original_image, 100, 200)
edges_noisy = canny_edge_detection(noisy_image, 100, 200)
edges_filtered = canny_edge_detection(filtered_image, 100, 200)

plot_canny()
plot_images()



from skimage.metrics import structural_similarity as ssim

def edge_density(edge_image):
    # Edge pixels are white (255), so we count these
    return np.sum(edge_image == 255) / edge_image.size

def compare_edges(edge1, edge2):
    # Calculate the Structural Similarity Index (SSIM) for edges
    return ssim(edge1, edge2, data_range=edge1.max() - edge1.min())



# Calculate edge densities
density_original = edge_density(edges_original)
density_noisy = edge_density(edges_noisy)
density_filtered = edge_density(edges_filtered)

# Calculate SSIM between edge maps
ssim_original_noisy = compare_edges(edges_original, edges_noisy)
ssim_original_filtered = compare_edges(edges_original, edges_filtered)

print(f"Edge Density - Original: {density_original}")
print(f"Edge Density - Noisy: {density_noisy}")
print(f"Edge Density - Filtered: {density_filtered}")
print(f"SSIM (Original vs Noisy): {ssim_original_noisy}")
print(f"SSIM (Original vs Filtered): {ssim_original_filtered}")
