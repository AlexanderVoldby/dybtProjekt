import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# ===========================
# 0. Configuration
# ===========================

MODEL_ARCHITECTURE = 'resnet' # Choose between 'cnn' and 'resnet'
datasplit = 'random' # Choose between 'random' and 'models'

# Paths to the dataset directories
# data_dir = f'/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/split_{datasplit}/'
data_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/sd2.1/314'

# Path to the saved model
if MODEL_ARCHITECTURE == 'cnn':
    model_path = f'/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/simple_cnn_{datasplit}.pth'
    gradcam_base_dir = f'gradcam_outputs/cnn_{datasplit}_heatmaps'

elif MODEL_ARCHITECTURE == 'resnet':
    model_path = f'/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/resnet_both_dd_sd.pth'
    gradcam_base_dir = f'gradcam_outputs/resnetboth_{datasplit}_heatmaps_sd'





real_test_dir = os.path.join(data_dir, 'real', 'test')
synthetic_test_dir = os.path.join(data_dir, 'synthetic', 'test')

gradcam_real_dir = os.path.join(gradcam_base_dir, 'real_heatmaps')
gradcam_synthetic_dir = os.path.join(gradcam_base_dir, 'synthetic_heatmaps')

# Verify that all directories and model exist
for dir_path in [real_test_dir, synthetic_test_dir]:
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")

if not os.path.isfile(model_path):
    raise ValueError(f"Model file does not exist: {model_path}")

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

# ===========================
# 6. Main Function
# ===========================
def main():

    # ===========================
    # 6.2 Define Data Transforms
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

    if MODEL_ARCHITECTURE == 'cnn':
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif MODEL_ARCHITECTURE == 'resnet':
        # Initialize resnet18 model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: 'real' and 'synthetic'

        # Load the fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Unknown MODEL_ARCHITECTURE: {MODEL_ARCHITECTURE}")

    model.to(device)
    model.eval()

    # ===========================
    # 6.5 Initialize Grad-CAM
    # ===========================
    if MODEL_ARCHITECTURE == 'cnn':
        # Identify the target layer (last convolutional layer)
        # In SimpleCNN, the last Conv2d layer is features[8]
        target_layer = model.features[8]
    elif MODEL_ARCHITECTURE == 'resnet':
        # Identify the target layer (last convolutional layer in resnet18)
        # In resnet18, the last conv layer is layer4[1].conv2
        target_layer = model.layer4[1].conv2
    else:
        raise ValueError(f"Unknown MODEL_ARCHITECTURE: {MODEL_ARCHITECTURE}")

    grad_cam = GradCAM(model, target_layer)

    # ===========================
    # 6.6 Select Images for Visualization
    # ===========================
    # Define the number of samples to visualize from each class
    num_samples_per_class = int(1e100)  # Adjust as needed

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
    # 6.7 Apply Grad-CAM and Save Heatmaps
    # ===========================
    print("Applying Grad-CAM and generating heatmaps...")

    # Create the base and subdirectories if they don't exist
    os.makedirs(gradcam_real_dir, exist_ok=True)
    os.makedirs(gradcam_synthetic_dir, exist_ok=True)

    for idx in tqdm(selected_indices, desc="Generating Grad-CAM heatmaps"):
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

        # Get the original image size
        img_filename = test_dataset.image_paths[idx]
        original_image = Image.open(img_filename).convert('RGB')
        original_size = original_image.size  # (width, height)

        # Resize heatmap to match original image size
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_resized = heatmap_resized.resize(original_size, resample=Image.BILINEAR)

        # Save the grayscale heatmap image
        heatmap_grayscale = heatmap_resized  # Already grayscale

        # Determine save directory based on true label
        true_label = 'real' if label == 0 else 'synthetic'
        if true_label == 'real':
            save_subdir = gradcam_real_dir
        else:
            save_subdir = gradcam_synthetic_dir

        # Save the grayscale heatmap image
        img_basename = os.path.basename(img_filename)
        save_filename = f"gradcam_{img_basename}"
        save_path = os.path.join(save_subdir, save_filename)
        heatmap_grayscale.save(save_path)

        # Uncomment the following code to apply colormap and save colored heatmaps
        # # Apply colormap to heatmap
        # heatmap_colored = np.array(heatmap_resized)
        # colormap = plt.get_cmap('jet')
        # heatmap_colored = colormap(heatmap_colored / 255.0)[:, :, :3]
        # heatmap_colored = np.uint8(heatmap_colored * 255)
        # heatmap_colored_image = Image.fromarray(heatmap_colored)
        #
        # # Save the colored heatmap image
        # colored_save_filename = f"gradcam_colored_{img_basename}"
        # colored_save_path = os.path.join(save_subdir, colored_save_filename)
        # heatmap_colored_image.save(colored_save_path)

        # Uncomment the following code to overlay the heatmap on the original image
        # # Unnormalize the image for visualization
        # img_unnorm = unnormalize(img_tensor, imagenet_mean, imagenet_std)
        # # Resize the unnormalized image to original size
        # img_unnorm_resized = Image.fromarray((img_unnorm * 255).astype(np.uint8)).resize(original_size, resample=Image.BILINEAR)
        # # Overlay heatmap on image
        # overlayed_img = Image.blend(img_unnorm_resized, heatmap_colored_image, alpha=0.4)
        # # Save overlayed image
        # overlayed_img_save_path = save_path.replace("gradcam_", "overlay_")
        # overlayed_img.save(overlayed_img_save_path)

    # ===========================
    # 6.8 Cleanup
    # ===========================
    # Remove hooks
    grad_cam.remove_hooks()

    print("Grad-CAM heatmaps generated and saved.")

if __name__ == "__main__":
    main()