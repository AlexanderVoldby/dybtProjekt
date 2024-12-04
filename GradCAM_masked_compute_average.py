import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu  # Import Otsu's thresholding method
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

def compute_gradcam_outside_proportion(
    img_path, 
    heatmap_path, 
    use_otsu=True,  # Parameter to toggle Otsu's thresholding
    threshold_value=None  # Optional fixed threshold value
):
    """
    Computes the proportion of the Grad-CAM area that lies outside the main object in the image.

    Parameters:
        img_path (str): Path to the background-removed image (PNG with transparency).
        heatmap_path (str): Path to the saved Grad-CAM heatmap (grayscale image).
        use_otsu (bool): Whether to use Otsu's method for thresholding. Defaults to True.
        threshold_value (float, optional): Fixed threshold to binarize the Grad-CAM heatmap.
                                           Ignored if use_otsu is True.

    Returns:
        proportion (float): Proportion of Grad-CAM area outside the main object.
    """
    # ===========================
    # Step 1: Load the Background-Removed Image
    # ===========================
    try:
        img = Image.open(img_path).convert('RGBA')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None
    img_np = np.array(img)

    # Extract RGB channels and normalize to [0, 1]
    original_img = img_np[:, :, :3] / 255.0

    # ===========================
    # Step 2: Create the Main Object Mask
    # ===========================
    alpha_channel = img_np[:, :, 3]
    main_object_mask = (alpha_channel > 0).astype(np.uint8)

    # ===========================
    # Step 3: Load the Grad-CAM Heatmap
    # ===========================
    try:
        heatmap = Image.open(heatmap_path).convert('L')
    except Exception as e:
        print(f"Error loading heatmap {heatmap_path}: {e}")
        return None
    heatmap_np = np.array(heatmap) / 255.0  # Normalize to [0, 1]

    # Resize heatmap to match image size if necessary
    if heatmap_np.shape != main_object_mask.shape:
        heatmap = heatmap.resize((main_object_mask.shape[1], main_object_mask.shape[0]), Image.BILINEAR)
        heatmap_np = np.array(heatmap) / 255.0  # Normalize after resizing

    # ===========================
    # Step 4: Determine the Threshold
    # ===========================
    if use_otsu:
        try:
            threshold = threshold_otsu(heatmap_np)
            # Handle case where all values are the same
            if np.isnan(threshold):
                print(f"Otsu's threshold returned NaN for heatmap {heatmap_path}. Using threshold=0.5")
                threshold = 0.5
        except Exception as e:
            print(f"Error computing Otsu's threshold for heatmap {heatmap_path}: {e}")
            threshold = 0.5  # Default threshold
    elif threshold_value is not None:
        threshold = threshold_value
    else:
        raise ValueError("Either use_otsu must be True or a threshold_value must be provided.")

    # ===========================
    # Step 5: Create the Grad-CAM Mask by Thresholding
    # ===========================
    gradcam_mask = (heatmap_np >= threshold).astype(np.uint8)

    # ===========================
    # Step 6: Compute Areas and Proportion
    # ===========================
    total_gradcam_area = np.sum(gradcam_mask)
    outside_area = np.sum((gradcam_mask == 1) & (main_object_mask == 0))

    if total_gradcam_area == 0:
        proportion = 0.0
    else:
        proportion = outside_area / total_gradcam_area

    return proportion

def find_corresponding_image(image_filename, heatmap_dir, heatmap_prefix='gradcam_'):
    """
    Finds the corresponding heatmap file for a given image filename by matching
    image filenames (without spaces) to heatmap filenames (with spaces).

    Parameters:
        image_filename (str): Filename of the image (spaces removed).
        heatmap_dir (str): Directory where heatmaps are stored.
        heatmap_prefix (str): Prefix added to heatmap filenames.

    Returns:
        heatmap_path (str): Full path to the corresponding heatmap file.
                             Returns None if not found.
    """
    # Remove extension from image filename for matching
    image_name_core, image_ext = os.path.splitext(image_filename)

    # Iterate through all files in heatmap_dir to find a matching heatmap
    for h_filename in os.listdir(heatmap_dir):
        # Check if heatmap_prefix is present
        if heatmap_prefix and not h_filename.startswith(heatmap_prefix):
            continue  # Skip files that don't have the required prefix

        # Remove prefix
        if heatmap_prefix:
            h_filename_core = h_filename[len(heatmap_prefix):]
        else:
            h_filename_core = h_filename

        # Remove spaces and extension from heatmap filename
        h_name_no_space, h_ext = os.path.splitext(h_filename_core)
        h_name_no_space = h_name_no_space.replace(' ', '')

        # Compare to image filename without extension
        if h_name_no_space == image_name_core:
            return os.path.join(heatmap_dir, h_filename)

    # If no matching heatmap is found
    return None

def main():
    # ===========================
    # Configuration
    # ===========================
    # Directory containing background-removed images
    image_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/split_random/real/test'  # Update this path

    # Directory containing Grad-CAM heatmaps
    heatmap_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/cnn_random/real_heatmaps'  # Update this path

    # Heatmap filename prefix (if any)
    heatmap_prefix = 'gradcam_'  # Update if different

    # Whether to use Otsu's thresholding
    use_otsu = True

    # Fixed threshold value (if not using Otsu)
    fixed_threshold = None  # Set to a float value if use_otsu is False

    # Number of samples to plot
    num_samples_to_plot = 5

    # ===========================
    # Step 1: Gather Image and Heatmap Pairs
    # ===========================
    image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_filenames:
        print(f"No image files found in directory: {image_dir}")
        return

    proportions = []
    missing_heatmaps = []
    processed_pairs = []  # List to store tuples of (image_path, heatmap_path)

    for img_filename in tqdm(image_filenames, desc="Processing images"):
        img_path = os.path.join(image_dir, img_filename)
        heatmap_path = find_corresponding_image(img_filename, heatmap_dir, heatmap_prefix)

        if heatmap_path is None:
            print(f"Warning: No corresponding heatmap found for image {img_filename}. Skipping.")
            missing_heatmaps.append(img_filename)
            continue

        proportion = compute_gradcam_outside_proportion(
            img_path=img_path,
            heatmap_path=heatmap_path,
            use_otsu=use_otsu,
            threshold_value=fixed_threshold
        )

        if proportion is not None:
            proportions.append(proportion)
            processed_pairs.append((img_path, heatmap_path))  # Add to processed pairs
            # print(f"Processed {img_filename}: Proportion = {proportion:.4f}")
        else:
            print(f"Failed to process {img_filename} due to errors.")

    # ===========================
    # Step 2: Compute Average Proportion
    # ===========================
    if proportions:
        average_proportion = np.mean(proportions)
        print("\n==============================")
        print(f"Processed {len(proportions)} image-heatmap pairs.")
        if missing_heatmaps:
            print(f"Skipped {len(missing_heatmaps)} images due to missing heatmaps.")
        print(f"Average Proportion of Grad-CAM area outside the main object: {average_proportion:.4f}")
        print("==============================")
    else:
        print("No valid image-heatmap pairs were processed.")

    # ===========================
    # Step 3: Plot 5 Random Samples
    # ===========================
    if len(processed_pairs) < num_samples_to_plot:
        print(f"Not enough processed pairs to plot {num_samples_to_plot} samples. Available: {len(processed_pairs)}")
        return

    # Select 5 random samples
    random.seed(42)  # For reproducibility
    sampled_pairs = random.sample(processed_pairs, num_samples_to_plot)

    # Create a matplotlib figure
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 5 * num_samples_to_plot))

    if num_samples_to_plot == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D

    for idx, (img_path, heatmap_path) in enumerate(sampled_pairs):
        # Load image
        img = Image.open(img_path).convert('RGBA')
        img_np = np.array(img)
        original_img = img_np[:, :, :3] / 255.0  # Normalize

        # Load heatmap
        heatmap = Image.open(heatmap_path).convert('L')
        heatmap_np = np.array(heatmap) / 255.0  # Normalize

        # Resize heatmap to match image size if necessary
        if heatmap_np.shape[:2] != original_img.shape[:2]:
            heatmap = heatmap.resize((original_img.shape[1], original_img.shape[0]), Image.BILINEAR)
            heatmap_np = np.array(heatmap) / 255.0  # Normalize after resizing

        # Create heatmap overlay
        heatmap_colored = plt.cm.jet(heatmap_np)[:, :, :3]  # Apply colormap and remove alpha
        overlay = (original_img * 0.6 + heatmap_colored * 0.4)  # Adjust alpha as needed
        overlay = np.clip(overlay, 0, 1)  # Ensure values are within [0,1]

        # Plot original image
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f"Original Image {idx+1}")

        # Plot overlay
        axes[idx, 1].imshow(overlay)
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f"Heatmap Overlay {idx+1}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()