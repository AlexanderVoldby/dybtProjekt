import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from scipy import interpolate
from scipy.stats import linregress

def compute_rapsd(image):
    """
    Compute Radially Averaged Power Spectral Density (RAPSD) of a 2D image.
    
    Parameters:
        image (2D numpy array): Grayscale image.
        
    Returns:
        freqs (1D numpy array): Frequency values.
        rapsd (1D numpy array): Radially averaged power spectral density.
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    
    # Compute power spectrum
    power_spectrum = np.abs(fft_shift) ** 2
    
    # Get image dimensions
    y, x = power_spectrum.shape
    y_center, x_center = y // 2, x // 2
    
    # Create coordinate grids
    Y, X = np.ogrid[:y, :x]
    R = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    R = R.flatten()
    power = power_spectrum.flatten()
    
    # Define radial bins
    R = np.array([int(r) for r in R])
    max_radius = int(np.max(R))
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
    Load all images from a root folder, including all subfolders.
    
    Parameters:
        root_folder (str): Path to the root folder containing images in subfolders.
        resize_size (tuple): Desired image size (height, width).
        is_color (bool): Whether to load images in color.
        
    Returns:
        images (list of numpy arrays): List of loaded and preprocessed images.
        labels (list of str): List of image filenames.
    """
    images = []
    labels = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_path = os.path.join(subdir, file)
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
                    img_scaled = (img_resized - 0.5) * 2
                    
                    images.append(img_scaled)
                    labels.append(file)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return images, labels

def linear_fit_loglog(frequencies, rapsd):
    """
    Fit a line to the log-log RAPSD and return the slope and intercept.
    
    Parameters:
        frequencies (1D numpy array): Frequency values.
        rapsd (1D numpy array): RAPSD values.
        
    Returns:
        slope (float): Slope of the fitted line.
        intercept (float): Intercept of the fitted line.
    """
    # Exclude the first frequency (DC component) and any zero or negative values
    mask = (frequencies > 0) & (rapsd > 0)
    log_freq = np.log(frequencies[mask])
    log_power = np.log(rapsd[mask])
    
    # Perform linear regression
    slope, intercept, _, _, _ = linregress(log_freq, log_power)
    
    return slope, intercept

def main():
    # ===========================
    # 1. Specify Image Paths
    # ===========================
    # Replace these paths with the actual paths where your images are stored
    synthetic_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-synthetic-classwise-16'
    real_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-real-train-fewshot'
    
    # ===========================
    # 2. Load Images
    # ===========================
    print("Loading synthetic images...")
    synthetic_images, synthetic_labels = load_images_from_folder(
        synthetic_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(synthetic_images)} synthetic images.")
    
    print("Loading real images...")
    real_images, real_labels = load_images_from_folder(
        real_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(real_images)} real images.")
    
    # Combine all images and labels
    all_images = synthetic_images + real_images
    all_labels = synthetic_labels + real_labels
    
    # ===========================
    # 3. Compute RAPSD for All Images
    # ===========================
    print("Computing RAPSD for all images...")
    rapsd_list = []
    freq_list = []
    
    for idx, img in enumerate(all_images):
        freqs, rapsd = compute_rapsd(img)
        freq_list.append(freqs)
        rapsd_list.append(rapsd)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_images):
            print(f"Processed {idx + 1}/{len(all_images)} images.")
    
    # ===========================
    # 4. Separate Synthetic and Real Images
    # ===========================
    num_synthetic = len(synthetic_images)
    synthetic_indices = list(range(num_synthetic))
    real_indices = list(range(num_synthetic, len(all_images)))
    
    synthetic_rapsd = [rapsd_list[i] for i in synthetic_indices]
    real_rapsd = [rapsd_list[i] for i in real_indices]
    synthetic_freq = [freq_list[i] for i in synthetic_indices]
    real_freq = [freq_list[i] for i in real_indices]
    
    # ===========================
    # 5. Compute Average RAPSD
    # ===========================
    print("Computing average RAPSD for synthetic and real images...")
    # Ensure all RAPSD arrays have the same length
    min_length = min(
        min([len(r) for r in synthetic_rapsd]),
        min([len(r) for r in real_rapsd])
    )
    
    synthetic_rapsd_trimmed = [r[:min_length] for r in synthetic_rapsd]
    real_rapsd_trimmed = [r[:min_length] for r in real_rapsd]
    synthetic_freq_trimmed = [f[:min_length] for f in synthetic_freq]
    real_freq_trimmed = [f[:min_length] for f in real_freq]
    
    mean_rapsd_synthetic = np.mean(synthetic_rapsd_trimmed, axis=0)
    mean_rapsd_real = np.mean(real_rapsd_trimmed, axis=0)
    freqs = synthetic_freq_trimmed[0]  # Assuming all frequencies are the same after trimming
    
    # ===========================
    # 6. Plot Average RAPSD Comparison
    # ===========================
    print("Plotting average RAPSD comparison...")
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs[1:], mean_rapsd_synthetic[1:], label='Synthetic Images', marker='o', linestyle='-', markersize=4)
    plt.loglog(freqs[1:], mean_rapsd_real[1:], label='Real Images', marker='s', linestyle='-', markersize=4)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Average Radially Averaged Power Spectral Density (RAPSD)')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('average_rapsd_comparison.png', dpi=300)
    plt.show()
    
    # ===========================
    # 7. Fit Linear Models
    # ===========================
    print("Fitting linear models to average RAPSD curves...")
    slope_syn, intercept_syn = linear_fit_loglog(freqs, mean_rapsd_synthetic)
    slope_real, intercept_real = linear_fit_loglog(freqs, mean_rapsd_real)
    
    # ===========================
    # 8. Plot with Linear Fits
    # ===========================
    print("Plotting average RAPSD with linear fits...")
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs[1:], mean_rapsd_synthetic[1:], label='Synthetic Images', marker='o', linestyle='-', markersize=4)
    plt.loglog(freqs[1:], mean_rapsd_real[1:], label='Real Images', marker='s', linestyle='-', markersize=4)
    
    # Generate points for the fitted lines
    log_freq = np.log(freqs[1:])
    fitted_syn = np.exp(intercept_syn) * freqs[1:]**slope_syn
    fitted_real = np.exp(intercept_real) * freqs[1:]**slope_real
    
    plt.loglog(freqs[1:], fitted_syn, linestyle='--', color='blue', label=f'Synthetic Fit (slope={slope_syn:.2f})')
    plt.loglog(freqs[1:], fitted_real, linestyle='--', color='orange', label=f'Real Fit (slope={slope_real:.2f})')
    
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Average RAPSD with Linear Fits (Log-Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('average_rapsd_with_fits.png', dpi=300)
    plt.show()
    
    # ===========================
    # 9. Print Conclusions
    # ===========================
    print("\nSpectral Analysis Completed.")
    print(f"Synthetic Images - Average Slope: {slope_syn:.2f}")
    print(f"Real Images - Average Slope: {slope_real:.2f}")

if __name__ == "__main__":
    main()