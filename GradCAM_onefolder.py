import os
import copy
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore")

# ===========================
# 0. Configuration
# ===========================

MODEL_PATH = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/resnet_both_dd_sd.pth'            # Path to the trained model
IMAGE_FOLDER = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/sd2.1/314'       # Path to the folder containing input images
OUTPUT_FOLDER = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/gradcam_outputs/resnetboth_heatmaps_sd'          # Path to the folder to save heatmaps


# Specify the model architecture ('cnn' or 'resnet')
MODEL_ARCHITECTURE = 'resnet'  # Options: 'cnn' or 'resnet'

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
# 2. Define Simple CNN Model
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
# 3. Implement Grad-CAM
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

        # Interpolate to input size
        cam = torch.nn.functional.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)

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
# 4. Helper Functions
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

def get_model_and_target_layer(model_architecture, model_path, device):
    """
    Loads the specified model architecture and returns the model along with the target layer for Grad-CAM.

    Parameters:
        model_architecture (str): 'cnn' or 'resnet'.
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model on.

    Returns:
        model (nn.Module): Loaded model.
        target_layer (nn.Module): Target layer for Grad-CAM.
    """
    if model_architecture.lower() == 'cnn':
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        target_layer = model.features[-1]  # Last convolutional layer
    elif model_architecture.lower() == 'resnet':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes
        model.load_state_dict(torch.load(model_path, map_location=device))
        target_layer = model.layer4[-1].conv2  # Last convolutional layer in ResNet
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    model.to(device)
    model.eval()
    return model, target_layer

# ===========================
# 5. Main Function
# ===========================
def main():
    # ===========================
    # 5.1 Verify Paths
    # ===========================
    if not os.path.isfile(MODEL_PATH):
        raise ValueError(f"Model file does not exist: {MODEL_PATH}")
    if not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"Image folder does not exist: {IMAGE_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ===========================
    # 5.2 Define Data Transforms
    # ===========================
    # ImageNet statistics used during training
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transforms for testing
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # ===========================
    # 5.3 Initialize the Model and Grad-CAM
    # ===========================
    print("Initializing the model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, target_layer = get_model_and_target_layer(MODEL_ARCHITECTURE, MODEL_PATH, device)
    grad_cam = GradCAM(model, target_layer)

    # ===========================
    # 5.4 Collect Image Paths
    # ===========================
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')
    image_paths = [os.path.join(IMAGE_FOLDER, fname) for fname in os.listdir(IMAGE_FOLDER)
                   if fname.lower().endswith(supported_extensions)]

    if len(image_paths) == 0:
        raise ValueError(f"No supported image files found in {IMAGE_FOLDER}.")

    # ===========================
    # 5.5 Apply Grad-CAM and Save Heatmaps
    # ===========================
    print("Generating Grad-CAM heatmaps...")

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            input_tensor = test_transforms(image).unsqueeze(0).to(device)

            # Forward pass to get predictions
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = pred.item()

            # Generate heatmap
            heatmap = grad_cam.generate_heatmap(input_tensor, pred_class)

            # Resize heatmap to original image size
            original_size = image.size  # (width, height)
            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_size, resample=Image.BILINEAR)

            # Save the grayscale heatmap image
            img_basename = os.path.basename(img_path)
            save_filename = f"gradcam_{img_basename}"
            save_path = os.path.join(OUTPUT_FOLDER, save_filename)
            heatmap_resized.save(save_path)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    # ===========================
    # 5.6 Cleanup
    # ===========================
    grad_cam.remove_hooks()
    print(f"Grad-CAM heatmaps have been saved to {OUTPUT_FOLDER}.")

if __name__ == "__main__":
    main()