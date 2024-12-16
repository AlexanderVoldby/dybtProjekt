import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu  # Import Otsu's thresholding method


# Toggle between weighted and binary Grad-CAM activations
USE_WEIGHTED_GRADCAM = True  # Set to True for weighted, False for binary

# Paths to the image and its corresponding Grad-CAM heatmap
# img_nobackground_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/split_random/synthetic/test/AcuraIntegraTypeR2001_14.png'
img_nobackground_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/sd2.1/no-background-314/1_148.png'
# gradcam_heatmap_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/resnetboth_random_heatmaps/synthetic_heatmaps/gradcam_Acura Integra Type R 2001_14.png'
gradcam_heatmap_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/resnetboth_heatmaps_sd/gradcam_1_148.png'


def compute_gradcam_outside_proportion_with_visualization(
    img_path, 
    heatmap_path, 
    use_weighted=False,  # New parameter to toggle weighted vs binary
    use_otsu=True,       # Parameter to toggle Otsu's thresholding (used only if use_weighted=False)
    threshold_value=None,  # Optional fixed threshold value (used only if use_weighted=False and use_otsu=False)
    save_plots=False, 
    plot_save_path=None
):
    """
    Computes the proportion of the Grad-CAM area that lies outside the main object in the image
    and generates multiple plots to visualize each step.

    Parameters:
        img_path (str): Path to the background-removed image (PNG with transparency).
        heatmap_path (str): Path to the saved Grad-CAM heatmap (grayscale image).
        use_weighted (bool): Whether to use weighted Grad-CAM activations instead of binary.
        use_otsu (bool): Whether to use Otsu's method for thresholding. Defaults to True.
                          Ignored if use_weighted is True.
        threshold_value (float, optional): Fixed threshold to binarize the Grad-CAM heatmap.
                                           Ignored if use_weighted is True or use_otsu is True.
        save_plots (bool): Whether to save the generated plots.
        plot_save_path (str): Directory path to save the plots if save_plots is True.

    Returns:
        proportion (float): Proportion of Grad-CAM area outside the main object.
                            Computed as either:
                            - (Sum of activations outside) / (Sum of all activations) if use_weighted=True
                            - (Number of pixels above threshold outside) / (Number of pixels above threshold) if use_weighted=False
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
    # Step 4: Compute Proportion Based on Toggle
    # ===========================
    if use_weighted:
        total_gradcam_area = np.sum(heatmap_np)
        if total_gradcam_area == 0:
            proportion = 0.0
        else:
            outside_area = np.sum(heatmap_np * (main_object_mask == 0))
            proportion = outside_area / total_gradcam_area
        threshold_method = "Weighted Grad-CAM"
    else:
        # Determine the Threshold
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
            threshold_method = "Otsu's Threshold"
        elif threshold_value is not None:
            threshold = threshold_value
            threshold_method = f"Fixed Threshold ({threshold})"
        else:
            raise ValueError("Either use_otsu must be True or a threshold_value must be provided when use_weighted=False.")

        # Create the Grad-CAM Mask by Thresholding
        gradcam_mask = (heatmap_np >= threshold).astype(np.uint8)

        # Compute Areas and Proportion
        total_gradcam_area = np.sum(gradcam_mask)
        if total_gradcam_area == 0:
            proportion = 0.0
        else:
            outside_area = np.sum((gradcam_mask == 1) & (main_object_mask == 0))
            proportion = outside_area / total_gradcam_area

    # ===========================
    # Step 5: Visualization
    # ===========================
    if use_weighted:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Grad-CAM Analysis\nProportion Outside Main Object: {proportion:.2f}", fontsize=22)

    # Plotting Index Tracker
    plot_idx = 0

    # Subplot 1: Original Image
    axes_flat = axes.flatten()
    axes_flat[0].imshow(original_img)
    axes_flat[0].set_title('Original Image', fontsize=20)
    axes_flat[0].axis('off')

    # Subplot 2: Main Object Mask
    axes_flat[1].imshow(main_object_mask, cmap='gray')
    axes_flat[1].set_title('Car Mask', fontsize = 20)
    axes_flat[1].axis('off')

    if use_weighted:
        # Subplot 3: Overlay of Weighted Grad-CAM on Original Image
        heatmap_color = plt.get_cmap('jet')(heatmap_np)[:, :, :3]  # Remove alpha channel
        overlay = (original_img * 0.6 + heatmap_color * 0.4)  # Adjust alpha as needed
        overlay = np.clip(overlay, 0, 1)  # Ensure values are within [0,1]
        axes_flat[2].imshow(overlay)
        axes_flat[2].set_title('Grad-CAM on Original Image', fontsize=20)
        axes_flat[2].axis('off')

        # Subplot 4: Weighted Grad-CAM Areas Outside the Main Object
        outside_mask_weighted = heatmap_np * (main_object_mask == 0)
        axes_flat[3].imshow(outside_mask_weighted, cmap='jet')
        axes_flat[3].set_title('Grad-CAM Outside Car', fontsize=20)
        axes_flat[3].axis('off')
    else:
        # Subplot 3: Grad-CAM Heatmap
        axes_flat[2].imshow(heatmap_np, cmap='jet')
        axes_flat[2].set_title('Grad-CAM Heatmap', fontsize=20)
        axes_flat[2].axis('off')

        # Subplot 4: Thresholded Grad-CAM Mask
        axes_flat[3].imshow(gradcam_mask, cmap='gray')
        axes_flat[3].set_title(f'Grad-CAM Mask ({threshold_method})', fontsize=20)
        axes_flat[3].axis('off')

        # Subplot 5: Overlay of Thresholded Grad-CAM on Original Image
        heatmap_color = plt.get_cmap('jet')(gradcam_mask)[:, :, :3]  # Apply colormap to mask
        overlay = (original_img * 0.6 + heatmap_color * 0.4)  # Adjust alpha as needed
        overlay = np.clip(overlay, 0, 1)  # Ensure values are within [0,1]
        axes_flat[4].imshow(overlay)
        axes_flat[4].set_title('Thresholded Grad-CAM on Original Image', fontsize=20)
        axes_flat[4].axis('off')

        # Subplot 6: Grad-CAM Areas Outside the Main Object
        outside_mask = ((gradcam_mask == 1) & (main_object_mask == 0)).astype(np.uint8)
        axes_flat[5].imshow(outside_mask, cmap='gray')
        axes_flat[5].set_title('Grad-CAM Outside Car', fontsize=20)
        axes_flat[5].axis('off')

    plt.tight_layout()#rect=[0, 0.03, 1, 0.95])

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
    # plt.show()

    # ===========================
    # Step 6: Return the Proportion
    # ===========================
    return proportion


# Example usage:
if __name__ == "__main__":
    # ===========================
    # Configuration
    # ===========================

    # Ensure that the paths exist
    if not os.path.isfile(img_nobackground_path):
        raise FileNotFoundError(f"Image file not found: {img_nobackground_path}")
    if not os.path.isfile(gradcam_heatmap_path):
        raise FileNotFoundError(f"Grad-CAM heatmap file not found: {gradcam_heatmap_path}")

    # Compute proportion and visualize
    proportion = compute_gradcam_outside_proportion_with_visualization(
        img_path=img_nobackground_path,
        heatmap_path=gradcam_heatmap_path,
        use_weighted=USE_WEIGHTED_GRADCAM,  # Toggle usage here
        use_otsu=True if not USE_WEIGHTED_GRADCAM else False,  # Use Otsu only if not weighted
        threshold_value=None,  # Set a fixed threshold if not using Otsu and not weighted
        save_plots=True,  # Set to True to save the plots
        plot_save_path='/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/visualizations'  # Update as needed
    )

    print(f"Proportion of Grad-CAM area outside the main object: {proportion:.2f}")