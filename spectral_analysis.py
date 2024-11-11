import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import io, color, transform
import cv2
import os
import itertools

# %% [markdown]
# ## 1. Define RAPSD Function

# %%
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

# %% [markdown]
# ## 2. Load Images

# %%
def load_images_from_folder(folder, num_images=3, resize_size=(256, 256), is_color=True):
    """
    Load images from a specified folder.
    
    Parameters:
        folder (str): Path to the folder containing images.
        num_images (int): Number of images to load.
        resize_size (tuple): Desired image size (height, width).
        is_color (bool): Whether to load images in color.
        
    Returns:
        images (list of numpy arrays): List of loaded and preprocessed images.
        labels (list of str): List of image filenames.
    """
    images = []
    labels = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in itertools.islice((f for f in os.listdir(folder) if f.lower().endswith(supported_formats)), num_images):
        img_path = os.path.join(folder, filename)
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
        labels.append(filename)
    
    return images, labels

# %% [markdown]
# ## 3. Specify Image Paths

# %%
# Replace these paths with the actual paths where your images are stored
synthetic_folder = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-synthetic-classwise-16/synthetic_16/Acura Integra Type R 2001/'
real_folder = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-real-train-fewshot/Acura Integra Type R 2001/'

# Number of images to load from each folder
num_synthetic = 3
num_real = 3

# Load synthetic images
synthetic_images, synthetic_labels = load_images_from_folder(
    synthetic_folder, num_images=num_synthetic, resize_size=(256, 256), is_color=False
)

# Load real images
real_images, real_labels = load_images_from_folder(
    real_folder, num_images=num_real, resize_size=(256, 256), is_color=False
)

# Combine all images and labels
all_images = synthetic_images + real_images
all_labels = synthetic_labels + real_labels

# %% [markdown]
# ## 4. Compute Spectra

# %%
def compute_spectra(images):
    """
    Compute magnitude and phase spectra for a list of images.
    
    Parameters:
        images (list of 2D numpy arrays): List of grayscale images.
        
    Returns:
        spectra (list of 2D numpy arrays): List of Fourier transforms.
        magnitudes (list of 2D numpy arrays): List of magnitude spectra.
        phases (list of 2D numpy arrays): List of phase spectra.
    """
    spectra = []
    magnitudes = []
    phases = []
    
    for img in images:
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        spectra.append(fft_shift)
        magnitudes.append(magnitude)
        phases.append(phase)
    
    return spectra, magnitudes, phases

spectra, magnitudes, phases = compute_spectra(all_images)

# %% [markdown]
# ## 5. Compute RAPSD for All Images

# %%
rapsd_list = []
freq_list = []

for img in all_images:
    freqs, rapsd = compute_rapsd(img)
    freq_list.append(freqs)
    rapsd_list.append(rapsd)

# %% [markdown]
# ## 6. Visualization

# %%
# Number of images
num_images = len(all_images)

# Define indices for synthetic and real images
synthetic_indices = list(range(num_synthetic))
real_indices = list(range(num_synthetic, num_synthetic + num_real))

# Create a figure for original images and their spectra
fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

if num_images == 1:
    axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D

for idx in range(num_images):
    # Determine image type
    if idx in synthetic_indices:
        img_type = 'Synthetic'
    else:
        img_type = 'Real'
    
    # Original Image
    ax_orig = axes[idx, 0]
    ax_orig.imshow((all_images[idx] + 1) / 2, cmap='gray', vmin=0, vmax=1)
    ax_orig.set_title(f"{img_type}: {all_labels[idx]} - Original")
    ax_orig.axis('off')
    
    # Magnitude Spectrum
    ax_mag = axes[idx, 1]
    magnitude_display = np.log(magnitudes[idx] + 1e-5)
    im_mag = ax_mag.imshow(magnitude_display, cmap='viridis')
    ax_mag.set_title(f"{img_type}: {all_labels[idx]} - Magnitude Spectrum")
    ax_mag.axis('off')
    fig.colorbar(im_mag, ax=ax_mag, fraction=0.046, pad=0.04)
    
    # Phase Spectrum
    ax_phase = axes[idx, 2]
    im_phase = ax_phase.imshow(phases[idx], cmap='RdBu')
    ax_phase.set_title(f"{img_type}: {all_labels[idx]} - Phase Spectrum")
    ax_phase.axis('off')
    fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Enhanced RAPSD Comparison

# %%
plt.figure(figsize=(12, 8))

# Plot synthetic images RAPSD
for idx in synthetic_indices:
    plt.loglog(freq_list[idx][1:], rapsd_list[idx][1:], label=f"Synthetic: {all_labels[idx]}", marker='o')

# Plot real images RAPSD
for idx in real_indices:
    plt.loglog(freq_list[idx][1:], rapsd_list[idx][1:], label=f"Real: {all_labels[idx]}", marker='s')

plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Radially Averaged Power Spectral Density (RAPSD) Comparison')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# %% [markdown]
# ### Average RAPSDs for Synthetic and Real Images

# %%
# Separate synthetic and real images
synthetic_rapsd = [rapsd_list[i] for i in synthetic_indices]
real_rapsd = [rapsd_list[i] for i in real_indices]
synthetic_freq = [freq_list[i] for i in synthetic_indices]
real_freq = [freq_list[i] for i in real_indices]

# Compute mean RAPSD in log domain for synthetic images
mean_log_rapsd_synthetic = np.mean([np.log(s + 1e-30) for s in synthetic_rapsd], axis=0)
mean_log_rapsd_real = np.mean([np.log(s + 1e-30) for s in real_rapsd], axis=0)

# Plot average RAPSDs
plt.figure(figsize=(10, 6))
plt.loglog(synthetic_freq[0][1:], np.exp(mean_log_rapsd_synthetic)[1:], label='Synthetic Images', marker='o', linestyle='-')
plt.loglog(real_freq[0][1:], np.exp(mean_log_rapsd_real)[1:], label='Real Images', marker='s', linestyle='-')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Average RAPSD Comparison')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# %% [markdown]
# ### Linear Fit on Log-Log RAPSD

# %%
def linear_fit_loglog(frequencies, rapsd):
    """
    Fit a line to the log-log RAPSD and return the slope.
    
    Parameters:
        frequencies (1D numpy array): Frequency values.
        rapsd (1D numpy array): RAPSD values.
        
    Returns:
        slope (float): Slope of the fitted line.
        intercept (float): Intercept of the fitted line.
        equally_spaced_log_freq (1D numpy array): Equally spaced log frequencies.
        interpolated_log_power (1D numpy array): Interpolated log RAPSD.
    """
    # Exclude the first frequency (DC component)
    log_freq = np.log(frequencies[1:])
    log_power = np.log(rapsd[1:])
    
    # Define equally spaced log frequencies
    equally_spaced_log_freq = np.linspace(log_freq.min(), log_freq.max(), 1000)
    interpolated_log_power = np.interp(equally_spaced_log_freq, log_freq, log_power)
    
    # Perform linear regression
    slope, intercept = np.polyfit(equally_spaced_log_freq, interpolated_log_power, 1)
    
    return slope, intercept, equally_spaced_log_freq, interpolated_log_power

# Fit for synthetic images
slope_syn, intercept_syn, eq_log_freq_syn, interp_log_power_syn = linear_fit_loglog(
    synthetic_freq[0], np.exp(mean_log_rapsd_synthetic)
)

# Fit for real images
slope_real, intercept_real, eq_log_freq_real, interp_log_power_real = linear_fit_loglog(
    real_freq[0], np.exp(mean_log_rapsd_real)
)

# Plot with linear fits
plt.figure(figsize=(10, 6))
plt.loglog(synthetic_freq[0][1:], np.exp(mean_log_rapsd_synthetic)[1:], label='Synthetic Images', marker='o')
plt.loglog(real_freq[0][1:], np.exp(mean_log_rapsd_real)[1:], label='Real Images', marker='s')

# Plot linear fits
plt.loglog(
    np.exp(eq_log_freq_syn),
    np.exp(interp_log_power_syn),
    linestyle='--',
    color='blue',
    label=f'Synthetic Fit (slope={slope_syn:.2f})'
)
plt.loglog(
    np.exp(eq_log_freq_real),
    np.exp(interp_log_power_real),
    linestyle='--',
    color='orange',
    label=f'Real Fit (slope={slope_real:.2f})'
)

plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Average RAPSD with Linear Fits on Log-Log Scale')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# %% [markdown]
# ## 7. Conclusions

# %%
print("Spectral Analysis Completed.")
print(f"Synthetic Images - Average Slope: {slope_syn:.2f}")
print(f"Real Images - Average Slope: {slope_real:.2f}")