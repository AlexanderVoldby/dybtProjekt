import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_gradcam_outside_proportion(img_path, gradcam_heatmap_path, threshold=0.5):
    """
    Computes the proportion of the Grad-CAM area that lies outside the main object in the image.

    Parameters:
        img_path (str): Path to the background-removed image.
        gradcam_heatmap_path (str): Path to the saved Grad-CAM heatmap image.
        threshold (float): Threshold to binarize the Grad-CAM heatmap.

    Returns:
        proportion (float): Proportion of Grad-CAM area outside the main object.
    """
    # Load the background-removed image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)

    # Create the main object mask
    # Assuming background pixels are black (0, 0, 0)
    main_object_mask = np.any(img_np != [0, 0, 0], axis=-1).astype(np.uint8)

    # Load the Grad-CAM heatmap
    heatmap = Image.open(gradcam_heatmap_path).convert('L')
    heatmap_np = np.array(heatmap) / 255.0  # Normalize to [0, 1]

    # Resize heatmap to match image size if necessary
    if heatmap_np.shape != main_object_mask.shape:
        heatmap_np = np.array(heatmap.resize(main_object_mask.shape[::-1], Image.BILINEAR))

    # Create the Grad-CAM mask
    gradcam_mask = (heatmap_np >= threshold).astype(np.uint8)

    # Compute Grad-CAM area outside the main object
    outside_mask = ((gradcam_mask == 1) & (main_object_mask == 0)).astype(np.uint8)
    outside_area = np.sum(outside_mask)

    # Compute total Grad-CAM area
    total_gradcam_area = np.sum(gradcam_mask)

    # Handle the case where total_gradcam_area is zero
    if total_gradcam_area == 0:
        return 0.0

    # Compute the proportion
    proportion = outside_area / total_gradcam_area

    return proportion

# Example usage:
img_path = 'path_to_background_removed_image.png'
gradcam_heatmap_path = 'path_to_gradcam_heatmap.png'

proportion = compute_gradcam_outside_proportion(img_path, gradcam_heatmap_path, threshold=0.5)
print(f"Proportion of Grad-CAM area outside the main object: {proportion:.2f}")