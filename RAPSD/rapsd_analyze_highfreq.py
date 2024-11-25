import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from matplotlib.lines import Line2D

def compute_rapsd(image):
    """
    Compute Radially Averaged Power Spectral Density (RAPSD) of a 2D image.

    Parameters:
        image (2D numpy array): Grayscale image.

    Returns:
        freqs (1D numpy array): Frequency values.
        rapsd (1D numpy array): Radially averaged power spectral density.
    """
    # Compute 2D FFT and shift zero frequency to center
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    
    # Compute power spectrum
    power_spectrum = np.abs(fft_shift) ** 2
    
    # Image dimensions and center
    y, x = power_spectrum.shape
    y_center, x_center = y // 2, x // 2
    
    # Create coordinate grids
    Y, X = np.ogrid[:y, :x]
    R = np.sqrt((X - x_center)**2 + (Y - y_center)**2).flatten()
    power = power_spectrum.flatten()
    
    # Define radial bins
    R = R.astype(int)
    max_radius = R.max()
    radial_sum = np.bincount(R, weights=power)
    radial_count = np.bincount(R)
    radial_mean = radial_sum / radial_count
    
    # Frequencies corresponding to the radial bins
    freqs = np.fft.fftfreq(x, d=1.0)  # Assuming unit spacing
    freqs = np.fft.fftshift(freqs)
    freq_step = freqs[1] - freqs[0]
    freqs = np.arange(0, max_radius) * freq_step
    
    return freqs, radial_mean[:max_radius]

def load_images_from_folder(root_folder, resize_size=(256, 256), is_color=False):
    """
    Load all images directly from a root folder (no subfolders).

    Parameters:
        root_folder (str): Path to the root folder containing images.
        resize_size (tuple): Desired image size (height, width).
        is_color (bool): Whether to load images in color.

    Returns:
        images (list of numpy arrays): List of loaded and preprocessed images.
        filenames (list of str): List of image filenames.
    """
    images = []
    filenames = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    try:
        files = os.listdir(root_folder)
    except Exception as e:
        print(f"Error accessing folder {root_folder}: {e}")
        return images, filenames
    
    for file in files:
        if file.lower().endswith(supported_formats):
            img_path = os.path.join(root_folder, file)
            try:
                img = io.imread(img_path)
                
                # Convert to grayscale if needed
                if not is_color:
                    if img.ndim == 3:
                        img = color.rgb2gray(img)
                    else:
                        img = img.astype(np.float32)
                else:
                    if img.ndim == 2:
                        img = color.gray2rgb(img)
                
                # Resize
                img_resized = transform.resize(img, resize_size, anti_aliasing=True)
                
                # Convert to float32 and scale to [-1, 1]
                img_resized = img_resized.astype(np.float32)
                if is_color and img_resized.ndim == 3:
                    # Convert to grayscale by averaging channels
                    img_resized = color.rgb2gray(img_resized)
                img_scaled = (img_resized - 0.5) * 2  # Scale to [-1, 1]
                
                images.append(img_scaled)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return images, filenames

def main():
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # ===========================
    # 1. Specify Image Paths
    # ===========================
    # Training directories
    synthetic_train_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/synthetic/train'
    real_train_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/real/train'
    
    # ===========================
    # 2. Load Training Images
    # ===========================
    print("Loading synthetic training images...")
    synthetic_train_images, synthetic_train_filenames = load_images_from_folder(
        synthetic_train_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(synthetic_train_images)} synthetic training images.")
    
    print("Loading real training images...")
    real_train_images, real_train_filenames = load_images_from_folder(
        real_train_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(real_train_images)} real training images.")
    
    # ===========================
    # 3. Compute RAPSD for Training Images
    # ===========================
    print("Computing RAPSDs for synthetic training images...")
    synthetic_rapsds = []
    for img in synthetic_train_images:
        freqs, rapsd = compute_rapsd(img)
        synthetic_rapsds.append(rapsd[:len(freqs)])  # Ensure consistency
    synthetic_rapsds = np.array(synthetic_rapsds)
    
    print("Computing RAPSDs for real training images...")
    real_rapsds = []
    for img in real_train_images:
        freqs, rapsd = compute_rapsd(img)
        real_rapsds.append(rapsd[:len(freqs)])  # Ensure consistency
    real_rapsds = np.array(real_rapsds)
    
    # ===========================
    # 4. Compute Average RAPSDs
    # ===========================
    print("Computing average RAPSDs for synthetic and real training images...")
    mean_rapsd_synthetic = np.mean(synthetic_rapsds, axis=0)
    mean_rapsd_real = np.mean(real_rapsds, axis=0)
    
    # ===========================
    # 5. Plot RAPSDs with Averages
    # ===========================
    print("Plotting RAPSDs with average lines...")
    plt.figure(figsize=(12, 8))
    
    # Plot individual RAPSDs for synthetic images
    for rapsd in synthetic_rapsds:
        plt.loglog(freqs[1:], rapsd[1:], color='blue', alpha=0.1)
    
    # Plot individual RAPSDs for real images
    for rapsd in real_rapsds:
        plt.loglog(freqs[1:], rapsd[1:], color='orange', alpha=0.1)
    
    # Plot average RAPSD for synthetic images
    plt.loglog(freqs[1:], mean_rapsd_synthetic[1:], color='blue', linewidth=2.5, label='Synthetic Average RAPSD')
    
    # Plot average RAPSD for real images
    plt.loglog(freqs[1:], mean_rapsd_real[1:], color='orange', linewidth=2.5, label='Real Average RAPSD')
    
    # Plot aesthetics
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('RAPSD of Synthetic and Real Training Images with Averages')
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1, alpha=0.3, label='Synthetic Individual RAPSD'),
        Line2D([0], [0], color='orange', lw=1, alpha=0.3, label='Real Individual RAPSD'),
        Line2D([0], [0], color='blue', lw=2.5, label='Synthetic Average RAPSD'),
        Line2D([0], [0], color='orange', lw=2.5, label='Real Average RAPSD')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('average_rapsd_comparison.png', dpi=300)
    plt.show()
    
    # ===========================
    # 6. Identify Top 5 Real Images with Highest Average RAPSD
    # ===========================
    print("Identifying top 5 real images with highest average RAPSD...")
    # Compute average RAPSD per real image
    real_average_rapsd = np.mean(real_rapsds, axis=1)
    
    # Find indices of top 5 real images with highest average RAPSD
    top_5_indices = np.argsort(real_average_rapsd)[-5:][::-1]  # Descending order
    top_5_values = real_average_rapsd[top_5_indices]
    
    print("Top 5 real images with highest average RAPSD:")
    for i, idx in enumerate(top_5_indices):
        print(f"{i+1}. Filename: {real_train_filenames[idx]}, Average RAPSD: {top_5_values[i]:.2f}")
    
    # ===========================
    # 7. Plot Top 5 Real Images and Their RAPSDs
    # ===========================
    print("Plotting top 5 real images with highest average RAPSD and their RAPSD curves...")
    num_display = len(top_5_indices)
    fig, axes = plt.subplots(num_display, 2, figsize=(12, 4 * num_display))
    
    # If only one image, axes is not a 2D array
    if num_display == 1:
        axes = np.array([axes])
    
    for i, idx in enumerate(top_5_indices):
        # Retrieve the image and its RAPSD
        img = real_train_images[idx]
        filename = real_train_filenames[idx]
        freqs_img, rapsd_img = compute_rapsd(img)
        
        # Display the image
        ax_img = axes[i, 0]
        ax_img.imshow(img, cmap='gray')
        ax_img.axis('off')
        ax_img.set_title(f'Real Image {i+1}: {filename}\nAvg RAPSD: {top_5_values[i]:.2f}')
        
        # Display the RAPSD
        ax_rapsd = axes[i, 1]
        ax_rapsd.loglog(freqs_img[1:], rapsd_img[1:], color='green')
        ax_rapsd.set_xlabel('Frequency')
        ax_rapsd.set_ylabel('Power')
        ax_rapsd.set_title('RAPSD Curve')
        ax_rapsd.grid(True, which="both", ls="--", lw=0.5)
    
    plt.tight_layout()
    plt.savefig('top_5_high_average_rapsd_real_images.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()