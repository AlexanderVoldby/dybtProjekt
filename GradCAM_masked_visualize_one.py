import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu  # Import Otsu's thresholding method

def compute_gradcam_outside_proportion_with_visualization(
    img_path, 
    heatmap_path, 
    use_otsu=True,  # New parameter to toggle Otsu's thresholding
    threshold_value=None,  # Optional fixed threshold value
    save_plots=False, 
    plot_save_path=None
):
    """
    Computes the proportion of the Grad-CAM area that lies outside the main object in the image
    and generates multiple plots to visualize each step using Otsu's thresholding.

    Parameters:
        img_path (str): Path to the background-removed image (PNG with transparency).
        heatmap_path (str): Path to the saved Grad-CAM heatmap (grayscale image).
        use_otsu (bool): Whether to use Otsu's method for thresholding. Defaults to True.
        threshold_value (float, optional): Fixed threshold to binarize the Grad-CAM heatmap.
                                           Ignored if use_otsu is True.
        save_plots (bool): Whether to save the generated plots.
        plot_save_path (str): Directory path to save the plots if save_plots is True.

    Returns:
        proportion (float): Proportion of Grad-CAM area outside the main object.
    """
    # ===========================
    # Step 1: Load the Background-Removed Image
    # ===========================
    img = Image.open(img_path).convert('RGBA')
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
    heatmap = Image.open(heatmap_path).convert('L')
    heatmap_np = np.array(heatmap) / 255.0  # Normalize to [0, 1]

    # Resize heatmap to match image size if necessary
    if heatmap_np.shape != main_object_mask.shape:
        heatmap = heatmap.resize((main_object_mask.shape[1], main_object_mask.shape[0]), Image.BILINEAR)
        heatmap_np = np.array(heatmap) / 255.0  # Normalize after resizing

    # ===========================
    # Step 4: Determine the Threshold
    # ===========================
    if use_otsu:
        # Compute Otsu's threshold
        threshold = threshold_otsu(heatmap_np)
        threshold_method = "Otsu's Threshold"
    elif threshold_value is not None:
        # Use the provided fixed threshold
        threshold = threshold_value
        threshold_method = f"Fixed Threshold ({threshold})"
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

    # ===========================
    # Step 7: Visualization
    # ===========================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Grad-CAM Analysis\nProportion Outside Main Object: {proportion:.2f}", fontsize=16)

    # Subplot 1: Original Image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image with Background Removed')
    axes[0, 0].axis('off')

    # Subplot 2: Main Object Mask
    axes[0, 1].imshow(main_object_mask, cmap='gray')
    axes[0, 1].set_title('Main Object Mask')
    axes[0, 1].axis('off')

    # Subplot 3: Grad-CAM Heatmap
    axes[0, 2].imshow(heatmap_np, cmap='jet')
    axes[0, 2].set_title('Grad-CAM Heatmap')
    axes[0, 2].axis('off')

    # Subplot 4: Thresholded Grad-CAM Mask
    axes[1, 0].imshow(gradcam_mask, cmap='gray')
    axes[1, 0].set_title(f'Grad-CAM Mask ({threshold_method})')
    axes[1, 0].axis('off')

    # Subplot 5: Overlay of Grad-CAM on Original Image
    overlay = original_img.copy()
    heatmap_color = plt.get_cmap('jet')(heatmap_np)[:, :, :3]  # Remove alpha channel
    overlayed_img = heatmap_color * 0.4 + overlay
    overlayed_img = np.clip(overlayed_img, 0, 1)
    axes[1, 1].imshow(overlayed_img)
    axes[1, 1].set_title('Overlay of Grad-CAM on Original Image')
    axes[1, 1].axis('off')

    # Subplot 6: Grad-CAM Areas Outside the Main Object
    outside_mask = ((gradcam_mask == 1) & (main_object_mask == 0)).astype(np.uint8)
    axes[1, 2].imshow(outside_mask, cmap='gray')
    axes[1, 2].set_title('Grad-CAM Areas Outside Main Object')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot if required
    if save_plots:
        if plot_save_path is None:
            plot_save_path = os.path.dirname(heatmap_path)
        os.makedirs(plot_save_path, exist_ok=True)
        img_basename = os.path.basename(img_path)
        plot_filename = f"gradcam_visualization_{os.path.splitext(img_basename)[0]}.png"
        plot_full_path = os.path.join(plot_save_path, plot_filename)
        plt.savefig(plot_full_path, dpi=300)
        print(f"Visualization saved to {plot_full_path}")

    # Show the plot
    plt.show()

    # ===========================
    # Step 8: Return the Proportion
    # ===========================
    return proportion


# Example usage:
if __name__ == "__main__":
    img_nobackground_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/split_random_incomplete/synthetic/test/AcuraIntegraTypeR2001_14.png'
    gradcam_heatmap_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/simplecnn_random/synthetic_heatmaps/gradcam_Acura Integra Type R 2001_14.png'

    # Ensure that the paths exist
    if not os.path.isfile(img_nobackground_path):
        raise FileNotFoundError(f"Image file not found: {img_nobackground_path}")
    if not os.path.isfile(gradcam_heatmap_path):
        raise FileNotFoundError(f"Grad-CAM heatmap file not found: {gradcam_heatmap_path}")

    # Compute proportion and visualize using Otsu's thresholding
    proportion = compute_gradcam_outside_proportion_with_visualization(
        img_path=img_nobackground_path,
        heatmap_path=gradcam_heatmap_path,
        use_otsu=True,  # Enable Otsu's thresholding
        save_plots=True,  # Set to True to save the plots
        plot_save_path='/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/visualizations'  # Update as needed
    )

    print(f"Proportion of Grad-CAM area outside the main object: {proportion:.2f}")