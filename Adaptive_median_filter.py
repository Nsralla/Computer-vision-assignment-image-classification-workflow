import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from config import noise_name, noisy_image_path, original_image_path

def adaptive_median_filter(img, max_window_size=7):
    # Convert image to int16 to prevent overflow
    img = img.astype(np.int16)
    # Get image dimensions
    height, width = img.shape

    # Create a padded version of the image to handle borders
    pad_size = max_window_size // 2
    padded_img = np.pad(img, pad_size, mode='edge')

    # Initialize output image
    output_img = np.zeros_like(img)

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            window_size = 3  # Starting window size
            while True:
                # Define window boundaries
                half_size = window_size // 2
                x_min = i + pad_size - half_size
                x_max = i + pad_size + half_size + 1
                y_min = j + pad_size - half_size
                y_max = j + pad_size + half_size + 1

                # Extract the window
                window = padded_img[x_min:x_max, y_min:y_max]

                # Compute median, min, and max
                Z_med = np.median(window)
                Z_min = window.min()
                Z_max = window.max()
                Z_xy = padded_img[i + pad_size, j + pad_size]
                
                # Ensure variables are of int16
                Z_med = int(Z_med)
                Z_min = int(Z_min)
                Z_max = int(Z_max)
                Z_xy = int(Z_xy)

                A1 = Z_med - Z_min
                A2 = Z_med - Z_max

                if A1 > 0 and A2 < 0:
                    # Level B
                    B1 = Z_xy - Z_min
                    B2 = Z_xy - Z_max
                    if B1 > 0 and B2 < 0:
                        output_img[i, j] = Z_xy
                    else:
                        output_img[i, j] = Z_med
                    break
                else:
                    window_size += 2  # Increase window size
                    if window_size > max_window_size:
                        output_img[i, j] = Z_med
                        break
     # Convert output image back to uint8
    output_img = np.clip(output_img, 0, 255).astype(np.uint8) 
    print("Window Size:", window_size)         
    return output_img, window_size


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
    plt.title('low level gaussian Noisy Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Adaptive Median Filtered Image (Kernel Size = {kernel_size})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_canny():    
    # Display the Images and their Edges
    plt.figure(figsize=(18, 8))

    # Original Image and Edges
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(edges_original, cmap='gray')
    plt.title('Edges of Original Image')
    plt.axis('off')

    # Noisy Image and Edges
    plt.subplot(2, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('paper_salt_medium_noise Noise Grayscale Image')
    plt.axis('off')
    plt.subplot(2,3, 5)
    plt.imshow(edges_noisy, cmap='gray')
    plt.title('Edges of Noisy Image')
    plt.axis('off')

    # Filtered Image and Edges
    plt.subplot(2, 3, 3)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Adaptive Median Filtered Image (Kernel Size = {kernel_size})')
    plt.axis('off')
    plt.subplot(2, 3, 6)
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


start_time = time.time()  # Start timing
filtered_image, kernel_size = adaptive_median_filter(noisy_image)
filter_time = time.time() - start_time  # Calculate elapsed time

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

