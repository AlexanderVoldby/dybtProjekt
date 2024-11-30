import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random

# ===========================
# 1. Set Random Seeds for Reproducibility
# ===========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ===========================
# 2. Define Custom Dataset Class
# ===========================
class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from 'real' and 'synthetic' directories.
    Assigns label 0 for 'real' and 1 for 'synthetic'.
    """
    def __init__(self, real_dir, synthetic_dir, transform=None):
        """
        Initializes the dataset by listing all image paths and assigning labels.

        Parameters:
            real_dir (str): Path to the 'real' images.
            synthetic_dir (str): Path to the 'synthetic' images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Supported image extensions
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')

        # Collect 'real' images
        for root, _, files in os.walk(real_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(0)  # Label 0 for 'real'

        # Collect 'synthetic' images
        for root, _, files in os.walk(synthetic_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(1)  # Label 1 for 'synthetic'

        assert len(self.image_paths) == len(self.labels), "Mismatch between images and labels"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Label of the image (0: real, 1: synthetic).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and label -1 to indicate an error
            image = Image.new('RGB', (224, 224))
            label = -1

        if self.transform:
            image = self.transform(image)

        return image, label

# ===========================
# 3. Define Simple CNN Model
# ===========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3 (RGB), Output: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Output: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 28x28
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ===========================
# 4. Implement Grad-CAM
# ===========================
class GradCAM:
    """
    Grad-CAM implementation for visual explanations.
    """
    def __init__(self, model, target_layer):
        """
        Initializes Grad-CAM with the model and target layer.

        Parameters:
            model (nn.Module): Trained CNN model.
            target_layer (nn.Module): The layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks to capture activations and gradients.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register hooks
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_backward_hook(backward_hook)

        self.hook_handles.append(handle_forward)
        self.hook_handles.append(handle_backward)

    def generate_heatmap(self, input_tensor, target_class):
        """
        Generates Grad-CAM heatmap for a specific class.

        Parameters:
            input_tensor (Tensor): Preprocessed image tensor.
            target_class (int): The target class index.

        Returns:
            heatmap (ndarray): The generated heatmap.
        """
        # Ensure model is in evaluation mode
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        # Compute loss for the target class
        loss = output[0, target_class]
        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get captured gradients and activations
        gradients = self.gradients  # [batch_size, channels, H, W]
        activations = self.activations  # [batch_size, channels, H, W]

        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]

        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [batch_size, 1, H, W]

        # Apply ReLU
        cam = torch.relu(cam)

        # Normalize CAM to [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()

        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap

    def remove_hooks(self):
        """
        Removes the registered hooks.
        """
        for handle in self.hook_handles:
            handle.remove()

# ===========================
# 5. Helper Functions
# ===========================
def unnormalize(img_tensor, mean, std):
    """
    Unnormalizes an image tensor.

    Parameters:
        img_tensor (Tensor): Normalized image tensor.
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.

    Returns:
        img (ndarray): Unnormalized image as a numpy array.
    """
    img = img_tensor.clone().detach()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img

def overlay_heatmap(img, heatmap, alpha=0.4, cmap='jet'):
    """
    Overlays the heatmap on the image.

    Parameters:
        img (ndarray): Original image.
        heatmap (ndarray): Heatmap to overlay.
        alpha (float): Transparency for heatmap.
        cmap (str): Colormap for heatmap.

    Returns:
        overlayed_img (ndarray): Image with heatmap overlay.
    """
    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    try:
        # For Pillow >= 10.0.0
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        # For Pillow < 10.0.0
        resample_method = Image.LANCZOS

    heatmap = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]), resample_method)
    heatmap = np.array(heatmap)

    # Apply colormap
    heatmap = plt.get_cmap(cmap)(heatmap)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel

    # Overlay heatmap on image
    overlayed_img = heatmap * alpha + img
    overlayed_img = np.clip(overlayed_img, 0, 1)

    return overlayed_img

# ===========================
# 6. Main Function
# ===========================
def main():
    # ===========================
    # 6.1 Configuration and Paths
    # ===========================
    # Paths to the dataset directories
    data_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/split_models/'
    real_test_dir = os.path.join(data_dir, 'real', 'test')
    synthetic_test_dir = os.path.join(data_dir, 'synthetic', 'test')

    # Path to the saved model
    model_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/simple_cnn_models.pth'  # Update this path if necessary

    # Verify that all directories and model exist
    for dir_path in [real_test_dir, synthetic_test_dir]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")

    if not os.path.isfile(model_path):
        raise ValueError(f"Model file does not exist: {model_path}")

    # ===========================
    # 6.2 Define Data Transforms
    # ===========================
    # ImageNet statistics used during training
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transforms for testing
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # ===========================
    # 6.3 Load Test Dataset
    # ===========================
    print("Loading test datasets...")
    test_dataset = ImageDataset(real_dir=real_test_dir, 
                                synthetic_dir=synthetic_test_dir, 
                                transform=test_transforms)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # ===========================
    # 6.4 Initialize the Model
    # ===========================
    print("Initializing the model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ===========================
    # 6.5 Initialize Grad-CAM
    # ===========================
    # Identify the target layer (last convolutional layer)
    # In SimpleCNN, the last Conv2d layer is features[8]
    target_layer = model.features[8]
    grad_cam = GradCAM(model, target_layer)

    # ===========================
    # 6.6 Select Images for Visualization
    # ===========================
    # Define the number of samples to visualize from each class
    num_samples_per_class = 1e100  # Adjust as needed

    # Collect image indices for 'real' and 'synthetic'
    real_indices = [i for i, label in enumerate(test_dataset.labels) if label == 0]
    synthetic_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1]

    # Ensure there are enough samples
    num_real = min(num_samples_per_class, len(real_indices))
    num_synthetic = min(num_samples_per_class, len(synthetic_indices))
    if num_real == 0 or num_synthetic == 0:
        raise ValueError("Not enough samples in one of the classes to visualize.")

    # Randomly select samples
    selected_real_indices = random.sample(real_indices, num_real)
    selected_synthetic_indices = random.sample(synthetic_indices, num_synthetic)

    selected_indices = selected_real_indices + selected_synthetic_indices

    # ===========================
    # 6.7 Apply Grad-CAM and Visualize
    # ===========================
    print("Applying Grad-CAM and generating visualizations...")

    # Define output directories for real and synthetic
    gradcam_base_dir = 'gradcam_outputs'
    gradcam_real_dir = os.path.join(gradcam_base_dir, 'real_cnn_models')
    gradcam_synthetic_dir = os.path.join(gradcam_base_dir, 'synthetic_cnn_models')

    # Create the base and subdirectories if they don't exist
    os.makedirs(gradcam_real_dir, exist_ok=True)
    os.makedirs(gradcam_synthetic_dir, exist_ok=True)

    for idx in selected_indices:
        img_tensor, label = test_dataset[idx]
        if label == -1:
            print(f"Skipping image at index {idx} due to loading error.")
            continue

        # Add batch dimension and move to device
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_class = pred.item()

        # Generate heatmap for the predicted class
        heatmap = grad_cam.generate_heatmap(input_tensor, pred_class)

        # Unnormalize the image for visualization
        img_unnorm = unnormalize(img_tensor, imagenet_mean, imagenet_std)

        # Generate heatmap overlay
        overlayed_img = overlay_heatmap(img_unnorm, heatmap, alpha=0.4, cmap='jet')

        # Prepare title
        true_label = 'real' if label == 0 else 'synthetic'
        predicted_label = 'real' if pred_class == 0 else 'synthetic'
        title = f"True: {true_label} | Predicted: {predicted_label}"

        # Plot and save the image
        plt.figure(figsize=(8, 8))
        plt.imshow(overlayed_img)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        # Determine save directory based on true label
        if true_label == 'real':
            save_subdir = gradcam_real_dir
        else:
            save_subdir = gradcam_synthetic_dir

        # Save the figure in the corresponding subdirectory
        img_filename = test_dataset.image_paths[idx]
        img_basename = os.path.basename(img_filename)
        # Optionally, add a prefix to indicate Grad-CAM
        save_filename = f"gradcam_{img_basename}"
        save_path = os.path.join(save_subdir, save_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Grad-CAM visualization saved to {save_path}")

    # ===========================
    # 6.8 Cleanup
    # ===========================
    # Remove hooks
    grad_cam.remove_hooks()

    print("Grad-CAM visualizations completed.")

if __name__ == "__main__":
    main()