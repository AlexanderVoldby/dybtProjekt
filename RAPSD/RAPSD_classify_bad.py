import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
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
    Load all images directly from a root folder (no subfolders).
    
    Parameters:
        root_folder (str): Path to the root folder containing images.
        resize_size (tuple): Desired image size (height, width).
        is_color (bool): Whether to load images in color.
        
    Returns:
        images (list of numpy arrays): List of loaded and preprocessed images.
        labels (list of str): List of image filenames.
    """
    images = []
    labels = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    try:
        files = os.listdir(root_folder)
    except Exception as e:
        print(f"Error accessing folder {root_folder}: {e}")
        return images, labels
    
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

def classify_image(slope, intercept, slope_syn, intercept_syn, slope_real, intercept_real):
    """
    Classify an image based on the closest fitted line in (slope, intercept) space.
    
    Parameters:
        slope (float): Slope of the test image's fitted line.
        intercept (float): Intercept of the test image's fitted line.
        slope_syn (float): Slope of the synthetic average fitted line.
        intercept_syn (float): Intercept of the synthetic average fitted line.
        slope_real (float): Slope of the real average fitted line.
        intercept_real (float): Intercept of the real average fitted line.
        
    Returns:
        predicted_label (str): 'synthetic' or 'real' based on classification.
    """
    # Compute Euclidean distances in (slope, intercept) space
    distance_syn = np.sqrt((slope - slope_syn)**2 + (intercept - intercept_syn)**2)
    distance_real = np.sqrt((slope - slope_real)**2 + (intercept - intercept_real)**2)
    
    if distance_syn < distance_real:
        return 'synthetic'
    else:
        return 'real'

def main():
    # ===========================
    # 1. Specify Image Paths
    # ===========================
    # Training directories
    synthetic_train_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/synthetic/train'
    real_train_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/real/train'
    
    # Test directories
    synthetic_test_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/synthetic/test'
    real_test_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/real/test'
    
    # ===========================
    # 2. Load Training Images
    # ===========================
    print("Loading synthetic training images...")
    synthetic_train_images, synthetic_train_labels = load_images_from_folder(
        synthetic_train_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(synthetic_train_images)} synthetic training images.")
    
    print("Loading real training images...")
    real_train_images, real_train_labels = load_images_from_folder(
        real_train_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(real_train_images)} real training images.")
    
    # Combine training images and labels
    train_images = synthetic_train_images + real_train_images
    train_labels = ['synthetic'] * len(synthetic_train_images) + ['real'] * len(real_train_images)
    
    # ===========================
    # 3. Compute RAPSD for Training Images
    # ===========================
    print("Computing RAPSD for training images...")
    rapsd_list = []
    freq_list = []
    
    for idx, img in enumerate(train_images):
        freqs, rapsd = compute_rapsd(img)
        freq_list.append(freqs)
        rapsd_list.append(rapsd)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(train_images):
            print(f"Processed {idx + 1}/{len(train_images)} training images.")
    
    # ===========================
    # 4. Separate Synthetic and Real Training RAPSDs
    # ===========================
    num_synthetic_train = len(synthetic_train_images)
    synthetic_train_rapsd = rapsd_list[:num_synthetic_train]
    real_train_rapsd = rapsd_list[num_synthetic_train:]
    synthetic_train_freq = freq_list[:num_synthetic_train]
    real_train_freq = freq_list[num_synthetic_train:]
    
    # ===========================
    # 5. Compute Average RAPSD for Training
    # ===========================
    print("Computing average RAPSD for synthetic and real training images...")
    # Ensure all RAPSD arrays have the same length
    min_length_train = min(
        min([len(r) for r in synthetic_train_rapsd]),
        min([len(r) for r in real_train_rapsd])
    )
    
    synthetic_train_rapsd_trimmed = [r[:min_length_train] for r in synthetic_train_rapsd]
    real_train_rapsd_trimmed = [r[:min_length_train] for r in real_train_rapsd]
    synthetic_train_freq_trimmed = [f[:min_length_train] for f in synthetic_train_freq]
    real_train_freq_trimmed = [f[:min_length_train] for f in real_train_freq]
    
    mean_rapsd_synthetic = np.mean(synthetic_train_rapsd_trimmed, axis=0)
    mean_rapsd_real = np.mean(real_train_rapsd_trimmed, axis=0)
    freqs_train = synthetic_train_freq_trimmed[0]  # Assuming all frequencies are the same after trimming
    
    # ===========================
    # 6. Plot Average RAPSD Comparison for Training
    # ===========================
    print("Plotting average RAPSD comparison for training images...")
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_train[1:], mean_rapsd_synthetic[1:], label='Synthetic Training', marker='o', linestyle='-', markersize=4)
    plt.loglog(freqs_train[1:], mean_rapsd_real[1:], label='Real Training', marker='s', linestyle='-', markersize=4)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Average Radially Averaged Power Spectral Density (RAPSD) - Training')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('average_rapsd_training_comparison.png', dpi=300)
    plt.show()
    
    # ===========================
    # 7. Fit Linear Models to Training Average RAPSD
    # ===========================
    print("Fitting linear models to average RAPSD curves for training data...")
    slope_syn, intercept_syn = linear_fit_loglog(freqs_train, mean_rapsd_synthetic)
    slope_real, intercept_real = linear_fit_loglog(freqs_train, mean_rapsd_real)
    
    # ===========================
    # 8. Plot Training Average RAPSD with Fitted Lines
    # ===========================
    print("Plotting average RAPSD with fitted lines for training data...")
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs_train[1:], mean_rapsd_synthetic[1:], label='Synthetic Training', marker='o', linestyle='-', markersize=4)
    plt.loglog(freqs_train[1:], mean_rapsd_real[1:], label='Real Training', marker='s', linestyle='-', markersize=4)
    
    # Generate points for the fitted lines
    fitted_syn = np.exp(intercept_syn) * freqs_train**slope_syn
    fitted_real = np.exp(intercept_real) * freqs_train**slope_real
    
    plt.loglog(freqs_train[1:], fitted_syn[1:], linestyle='--', color='blue', label=f'Synthetic Fit (slope={slope_syn:.2f})')
    plt.loglog(freqs_train[1:], fitted_real[1:], linestyle='--', color='orange', label=f'Real Fit (slope={slope_real:.2f})')
    
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Average RAPSD with Linear Fits (Log-Log Scale) - Training Data')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('average_rapsd_training_with_fits.png', dpi=300)
    plt.show()
    
    # ===========================
    # 9. Load Test Images
    # ===========================
    print("Loading synthetic test images...")
    synthetic_test_images, synthetic_test_labels = load_images_from_folder(
        synthetic_test_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(synthetic_test_images)} synthetic test images.")
    
    print("Loading real test images...")
    real_test_images, real_test_labels = load_images_from_folder(
        real_test_root, resize_size=(256, 256), is_color=False
    )
    print(f"Loaded {len(real_test_images)} real test images.")
    
    # Combine test images and labels
    test_images = synthetic_test_images + real_test_images
    test_labels = ['synthetic'] * len(synthetic_test_images) + ['real'] * len(real_test_images)
    
    # ===========================
    # 10. Compute RAPSD for Test Images
    # ===========================
    print("Computing RAPSD for test images...")
    test_rapsd_list = []
    test_freq_list = []
    
    for idx, img in enumerate(test_images):
        freqs, rapsd = compute_rapsd(img)
        test_freq_list.append(freqs)
        test_rapsd_list.append(rapsd)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_images):
            print(f"Processed {idx + 1}/{len(test_images)} test images.")
    
    # ===========================
    # 11. Fit Linear Models to Test RAPSDs
    # ===========================
    print("Fitting linear models to test RAPSDs...")
    test_slopes = []
    test_intercepts = []
    
    for idx, (freqs, rapsd) in enumerate(zip(test_freq_list, test_rapsd_list)):
        # Trim to training RAPSD length
        rapsd_trimmed = rapsd[:min_length_train]
        freqs_trimmed = freqs[:min_length_train]
        slope, intercept = linear_fit_loglog(freqs_trimmed, rapsd_trimmed)
        test_slopes.append(slope)
        test_intercepts.append(intercept)
    
    # ===========================
    # 12. Classify Test Images
    # ===========================
    print("Classifying test images based on fitted lines...")
    predicted_labels = []
    
    for slope, intercept in zip(test_slopes, test_intercepts):
        pred_label = classify_image(
            slope, intercept,
            slope_syn, intercept_syn,
            slope_real, intercept_real
        )
        predicted_labels.append(pred_label)
    
    # ===========================
    # 13. Compute Test Accuracy
    # ===========================
    print("Computing test accuracy...")
    correct = 0
    for true_label, pred_label in zip(test_labels, predicted_labels):
        if true_label == pred_label:
            correct += 1
    accuracy = correct / len(test_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{len(test_labels)})")
    
    # ===========================
    # 14. Plot Test Classification Results (Optional)
    # ===========================
    # Visualize classification results in (slope, intercept) space
    plt.figure(figsize=(8, 6))
    # Plot training fits
    plt.scatter(slope_syn, intercept_syn, color='blue', marker='o', label='Synthetic Training Fit')
    plt.scatter(slope_real, intercept_real, color='orange', marker='s', label='Real Training Fit')
    # Plot test points
    for slope, intercept, true_label, pred_label in zip(test_slopes, test_intercepts, test_labels, predicted_labels):
        if true_label == 'synthetic':
            color = 'cyan' if pred_label == 'synthetic' else 'red'
            marker = 'o'
        else:
            color = 'gold' if pred_label == 'real' else 'blue'
            marker = 's'
        plt.scatter(slope, intercept, color=color, marker=marker, alpha=0.6)
    
    plt.xlabel('Slope')
    plt.ylabel('Intercept')
    plt.title('Test Image Classification in (Slope, Intercept) Space')
    plt.legend()
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig('test_classification_results.png', dpi=300)
    plt.show()
    
    # ===========================
    # 15. Print Conclusions
    # ===========================
    print("\nSpectral Analysis and Classification Completed.")
    print(f"Synthetic Training Images - Average Slope: {slope_syn:.2f}, Average Intercept: {intercept_syn:.2f}")
    print(f"Real Training Images - Average Slope: {slope_real:.2f}, Average Intercept: {intercept_real:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()