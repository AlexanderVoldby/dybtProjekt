import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import io, color, transform
import cv2
import os
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # Importing for dataset splitting

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

def sanitize_filename(filename):
    """
    Sanitize the filename by replacing path separators and special characters with underscores.

    Parameters:
        filename (str): Original filename.

    Returns:
        sanitized (str): Sanitized filename.
    """
    # Replace OS-specific path separators and other problematic characters
    sanitized = filename.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    # Remove or replace other unwanted characters
    sanitized = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in sanitized)
    return sanitized

def load_images_from_subfolders(input_root, num_images_per_folder=None, resize_size=(256, 256), is_color=False):
    """
    Recursively load all images from every subfolder within the input_root folder.

    Parameters:
        input_root (str): Path to the root folder containing subfolders with images.
        num_images_per_folder (int, optional): Number of images to load per subfolder. 
                                              If None, load all images.
        resize_size (tuple): Desired image size (height, width).
        is_color (bool): Whether to load images in color.

    Returns:
        images (list of numpy arrays): List of loaded and preprocessed images.
        labels (list of str): List of sanitized image filenames.
        image_types (list of str): List indicating 'Synthetic' or 'Real' for each image.
    """
    images = []
    labels = []
    image_types = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')

    for root, dirs, files in os.walk(input_root):
        # Determine if the current directory is synthetic or real based on the root path
        # Adjust this logic based on your directory naming conventions
        if 'synthetic' in root.lower():
            img_type = 'Synthetic'
        elif 'real' in root.lower():
            img_type = 'Real'
        else:
            img_type = 'Unknown'

        # Skip processing if the directory type is unknown
        if img_type == 'Unknown':
            continue

        for filename in itertools.islice((f for f in files if f.lower().endswith(supported_formats)), num_images_per_folder):
            img_path = os.path.join(root, filename)
            try:
                img = io.imread(img_path)
                
                # Convert to grayscale if needed
                if img.ndim == 3:
                    img = color.rgb2gray(img)
                elif img.ndim == 4:
                    img = color.rgb2gray(img.squeeze())
                else:
                    img = img.astype(np.float32)
                
                # Resize to specified size
                img_resized = transform.resize(img, resize_size, anti_aliasing=True)
                
                # Convert to float32 and scale to [-1, 1]
                img_resized = img_resized.astype(np.float32)
                if is_color and img_resized.ndim == 3:
                    # Convert to grayscale by averaging channels
                    img_resized = color.rgb2gray(img_resized)
                img_scaled = (img_resized - 0.5) * 2
                
                images.append(img_scaled)
                sanitized_label = sanitize_filename(os.path.relpath(img_path, input_root))
                labels.append(sanitized_label)
                image_types.append(img_type)
            except Exception as e:
                print(f"Failed to process image: {img_path}. Error: {e}")
                continue

    return images, labels, image_types

def save_spectrum(image_name, magnitude, phase, image_type, output_root, split):
    """
    Save the magnitude and phase spectra as image files in separate folders based on image type and split.

    Parameters:
        image_name (str): Original image filename.
        magnitude (2D numpy array): Magnitude spectrum.
        phase (2D numpy array): Phase spectrum.
        image_type (str): 'Synthetic' or 'Real'.
        output_root (str): Root path to save spectra.
        split (str): 'train' or 'test'.
    """
    # Define output folders based on split and image type
    magnitude_folder = os.path.join(output_root, split, image_type.lower(), 'magnitude')
    phase_folder = os.path.join(output_root, split, image_type.lower(), 'phase')

    # Ensure the directories exist
    os.makedirs(magnitude_folder, exist_ok=True)
    os.makedirs(phase_folder, exist_ok=True)

    # Prepare filenames
    base_name = sanitize_filename(os.path.splitext(image_name)[0])
    magnitude_filename = f"{base_name}_magnitude.png"
    phase_filename = f"{base_name}_phase.png"

    # Log scale for magnitude
    magnitude_display = np.log(magnitude + 1e-5)

    # Normalize magnitude and phase for saving
    mag_min, mag_max = magnitude_display.min(), magnitude_display.max()
    mag_norm = (magnitude_display - mag_min) / (mag_max - mag_min) if mag_max - mag_min != 0 else magnitude_display - mag_min
    mag_image = (mag_norm * 255).astype(np.uint8)

    phase_min, phase_max = phase.min(), phase.max()
    phase_norm = (phase - phase_min) / (phase_max - phase_min) if phase_max - phase_min != 0 else phase - phase_min
    phase_image = (phase_norm * 255).astype(np.uint8)

    # Save images using OpenCV
    cv2.imwrite(os.path.join(magnitude_folder, magnitude_filename), mag_image)
    cv2.imwrite(os.path.join(phase_folder, phase_filename), phase_image)

def visualize_spectra(all_images, all_labels, image_types, output_root, split, max_images_per_type=5):
    """
    Visualize and plot magnitude and phase spectra for a subset of images.

    Parameters:
        all_images (list): List of processed images.
        all_labels (list): List of sanitized image filenames.
        image_types (list): List indicating 'Synthetic' or 'Real' for each image.
        output_root (str): Root path where spectra are saved.
        split (str): 'train' or 'test'.
        max_images_per_type (int): Maximum number of images to visualize per type.
    """
    # Separate images by type
    synthetic_images = [(img, label) for img, label, typ in zip(all_images, all_labels, image_types) if typ == 'Synthetic']
    real_images = [(img, label) for img, label, typ in zip(all_images, all_labels, image_types) if typ == 'Real']

    # Limit the number of images to visualize
    synthetic_images = synthetic_images[:max_images_per_type]
    real_images = real_images[:max_images_per_type]

    total_plots = len(synthetic_images) + len(real_images)
    if total_plots == 0:
        print("No images to visualize.")
        return

    fig, axes = plt.subplots(total_plots, 3, figsize=(18, 6 * total_plots))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # If only one image, ensure axes are 2D
    if total_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    plot_idx = 0
    for img, label in synthetic_images:
        img_type = 'Synthetic'
        # Original Image
        ax_orig = axes[plot_idx, 0]
        ax_orig.imshow((img + 1) / 2, cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f"{img_type}: {label} - Original")
        ax_orig.axis('off')

        # Magnitude Spectrum
        ax_mag = axes[plot_idx, 1]
        mag_path = os.path.join(output_root, split, img_type.lower(), 'magnitude', f"{label}_magnitude.png")
        mag_image = io.imread(mag_path)
        ax_mag.imshow(mag_image, cmap='viridis')
        ax_mag.set_title(f"{img_type}: {label} - Magnitude Spectrum")
        ax_mag.axis('off')
        fig.colorbar(ax_mag.images[0], ax=ax_mag, fraction=0.046, pad=0.04)

        # Phase Spectrum
        ax_phase = axes[plot_idx, 2]
        phase_path = os.path.join(output_root, split, img_type.lower(), 'phase', f"{label}_phase.png")
        phase_image = io.imread(phase_path)
        ax_phase.imshow(phase_image, cmap='RdBu')
        ax_phase.set_title(f"{img_type}: {label} - Phase Spectrum")
        ax_phase.axis('off')
        fig.colorbar(ax_phase.images[0], ax=ax_phase, fraction=0.046, pad=0.04)

        plot_idx += 1

    for img, label in real_images:
        img_type = 'Real'
        # Original Image
        ax_orig = axes[plot_idx, 0]
        ax_orig.imshow((img + 1) / 2, cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f"{img_type}: {label} - Original")
        ax_orig.axis('off')

        # Magnitude Spectrum
        ax_mag = axes[plot_idx, 1]
        mag_path = os.path.join(output_root, split, img_type.lower(), 'magnitude', f"{label}_magnitude.png")
        mag_image = io.imread(mag_path)
        ax_mag.imshow(mag_image, cmap='viridis')
        ax_mag.set_title(f"{img_type}: {label} - Magnitude Spectrum")
        ax_mag.axis('off')
        fig.colorbar(ax_mag.images[0], ax=ax_mag, fraction=0.046, pad=0.04)

        # Phase Spectrum
        ax_phase = axes[plot_idx, 2]
        phase_path = os.path.join(output_root, split, img_type.lower(), 'phase', f"{label}_phase.png")
        phase_image = io.imread(phase_path)
        ax_phase.imshow(phase_image, cmap='RdBu')
        ax_phase.set_title(f"{img_type}: {label} - Phase Spectrum")
        ax_phase.axis('off')
        fig.colorbar(ax_phase.images[0], ax=ax_phase, fraction=0.046, pad=0.04)

        plot_idx += 1

    plt.tight_layout()
    plt.show()

def main():
    # Define paths directly here
    # Replace these paths with the actual paths on your system
    synthetic_folder = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-synthetic-classwise-16'
    real_folder = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-real-train-fewshot/'
    output_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/spectra_output/'

    # Number of images to load from each subfolder
    num_synthetic = None  # Set to None to load all images
    num_real = None       # Set to None to load all images

    # Define resize size
    resize_size = (256, 256)

    # Load synthetic images
    print("Loading synthetic images...")
    synthetic_images, synthetic_labels, synthetic_types = load_images_from_subfolders(
        synthetic_folder, num_images_per_folder=num_synthetic, resize_size=resize_size, is_color=False
    )

    # Load real images
    print("Loading real images...")
    real_images, real_labels, real_types = load_images_from_subfolders(
        real_folder, num_images_per_folder=num_real, resize_size=resize_size, is_color=False
    )

    # Combine all images, labels, and types
    all_images = synthetic_images + real_images
    all_labels = synthetic_labels + real_labels
    all_types = synthetic_types + real_types

    print(f"Total images loaded: {len(all_images)}")

    # Perform an 80-20 train-test split for each class to maintain class balance
    print("Splitting data into train and test sets...")

    # Separate indices for synthetic and real images
    synthetic_indices = [i for i, typ in enumerate(all_types) if typ == 'Synthetic']
    real_indices = [i for i, typ in enumerate(all_types) if typ == 'Real']

    # Perform split for synthetic images
    synthetic_train_idx, synthetic_test_idx = train_test_split(
        synthetic_indices, test_size=0.2, random_state=42, shuffle=True
    )

    # Perform split for real images
    real_train_idx, real_test_idx = train_test_split(
        real_indices, test_size=0.2, random_state=42, shuffle=True
    )

    # Combine train and test indices
    train_indices = synthetic_train_idx + real_train_idx
    test_indices = synthetic_test_idx + real_test_idx

    print(f"Training set size: {len(train_indices)}")
    print(f"Testing set size: {len(test_indices)}")

    # Define helper function to save spectra
    def save_spectra(indices, split):
        """
        Compute and save spectra for the given indices and split.

        Parameters:
            indices (list): List of indices to process.
            split (str): 'train' or 'test'.
        """
        for idx in tqdm(indices, desc=f"Processing {split} set"):
            img = all_images[idx]
            label = all_labels[idx]
            img_type = all_types[idx]

            # Compute magnitude and phase spectra
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            phase = np.angle(fft_shift)

            # Save spectra based on image type and split
            if img_type in ['Synthetic', 'Real']:
                save_spectrum(label, magnitude, phase, img_type, output_root, split)
            else:
                # If image type is unknown, skip saving
                print(f"Skipping image with unknown type: {label}")
                continue

    # Save spectra for training set
    print("Saving training spectra...")
    save_spectra(train_indices, 'train')

    # Save spectra for testing set
    print("Saving testing spectra...")
    save_spectra(test_indices, 'test')

    print("Spectral analysis and saving completed successfully.")

    # Optionally, visualize some spectra from train and test sets
    # Uncomment the lines below to visualize

    # print("Visualizing training spectra...")
    # visualize_spectra([all_images[i] for i in train_indices],
    #                  [all_labels[i] for i in train_indices],
    #                  [all_types[i] for i in train_indices],
    #                  output_root, 'train', max_images_per_type=5)

    # print("Visualizing testing spectra...")
    # visualize_spectra([all_images[i] for i in test_indices],
    #                  [all_labels[i] for i in test_indices],
    #                  [all_types[i] for i in test_indices],
    #                  output_root, 'test', max_images_per_type=5)

if __name__ == "__main__":
    main()