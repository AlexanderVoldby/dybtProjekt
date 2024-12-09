import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random

# ===========================
# 0. Toggle for Model Selection
# ===========================
# Choose between 'resnet18' and 'simplecnn'
MODEL_TYPE = 'simplecnn'  # Options: 'resnet18', 'simplecnn'

# Paths to the dataset directories
data_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/split_random'  # Update this path if necessary

if MODEL_TYPE == 'resnet18':
    model_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/resnet18_finetuned.pth'  # Update this path

elif MODEL_TYPE == 'simplecnn':
    model_path = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/simple_cnn_random.pth'  # Update this path

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
# 2. Define SimpleCNN Architecture
# ===========================
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for binary classification.
    This architecture should match exactly with the one used during training.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Convolutional Layer block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3 (RGB), Output: 32
            nn.BatchNorm2d(32),  # BatchNorm comes before ReLU
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 112x112

            # Convolutional Layer block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 56x56

            # Convolutional Layer block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Output: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 28x28
        )
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
# 3. Define Custom Dataset Class
# ===========================
class ImageDataset(torch.utils.data.Dataset):
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
        Retrieves the image, its label, and its path at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Label of the image (0: real, 1: synthetic).
            img_path (str): File path of the image.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Open image
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                # Convert to numpy array
                img_np = np.array(img)
                # Separate alpha channel
                alpha = img_np[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
                # Apply alpha channel to RGB channels
                img_np = img_np[:, :, :3]  # Get RGB channels
                img_np = img_np * np.expand_dims(alpha, axis=2)  # Apply alpha
                img_np = img_np.astype(np.uint8)  # Convert to uint8
                # Convert back to PIL Image
                img = Image.fromarray(img_np)
            else:
                # Ensure image is in RGB mode
                img = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and label -1 to indicate an error
            img = Image.new('RGB', (224, 224))
            label = -1

        if self.transform:
            img = self.transform(img)

        return img, label, img_path

def main():
    # ===========================
    # 4. Configuration and Hyperparameters
    # ===========================

    # Paths for 'real' and 'synthetic' test datasets
    real_test_dir = os.path.join(data_dir, 'real', 'test')
    synthetic_test_dir = os.path.join(data_dir, 'synthetic', 'test')

    # Verify that all directories exist
    for dir_path in [real_test_dir, synthetic_test_dir]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")

    # Hyperparameters
    batch_size = 32
    num_classes = 2  # 'real' and 'synthetic'

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================
    # 5. Data Transforms
    # ===========================
    # ImageNet statistics for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transforms for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # ===========================
    # 6. Load Test Dataset and DataLoader
    # ===========================
    print("Loading test dataset...")
    test_dataset = ImageDataset(real_dir=real_test_dir, 
                                synthetic_dir=synthetic_test_dir, 
                                transform=test_transform)

    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_size = len(test_dataset)
    print(f"Test dataset size: {dataset_size}")

    # Define class names
    class_names = ['real', 'synthetic']

    # ===========================
    # 7. Initialize the Model and Load Saved Weights
    # ===========================
    print("Initializing the model...")
    if MODEL_TYPE == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif MODEL_TYPE == 'simplecnn':
        # Initialize SimpleCNN
        model = SimpleCNN(num_classes=num_classes)

    else:
        raise ValueError("Unsupported MODEL_TYPE. Choose between 'resnet18' and 'simplecnn'.")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    # Load model weights
    state_dict = torch.load(model_path, map_location=device)

    if MODEL_TYPE == 'simplecnn':
        # If the model was trained using DataParallel, remove the 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Load the state_dict into the model
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"{MODEL_TYPE.capitalize()} model loaded successfully from {model_path}.")

    # ===========================
    # 8. Define Loss Function
    # ===========================
    criterion = nn.CrossEntropyLoss()

    # ===========================
    # 9. Evaluation on Test Set
    # ===========================
    print("Starting evaluation on the test set...")
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_image_paths = []

    with torch.no_grad():
        for inputs, labels, paths in tqdm(dataloader_test, desc="Testing Phase"):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Accumulate loss and correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Collect all predictions, labels, and image paths for metrics and visualization
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_paths.extend(paths)

    # Compute average loss and accuracy
    test_loss = running_loss / dataset_size
    test_acc = running_corrects.float() / dataset_size

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

    # ===========================
    # 10. Confusion Matrix and Classification Report
    # ===========================
    print("Generating confusion matrix and classification report...")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix on Test Set ({MODEL_TYPE.capitalize()})')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_set_evaluation.png', dpi=300)
    plt.show()

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)

    # ===========================
    # 11. (Optional) Visualize Sample Predictions with Balanced Classes
    # ===========================
    print("Visualizing balanced test predictions...")

    # Define the number of samples per class to plot
    num_samples_per_class = 3  # Total samples = 6

    # Find indices for each class
    real_indices = [i for i, label in enumerate(all_labels) if label == 0]
    synthetic_indices = [i for i, label in enumerate(all_labels) if label == 1]

    # Check if there are enough samples in each class
    if len(real_indices) < num_samples_per_class or len(synthetic_indices) < num_samples_per_class:
        raise ValueError("Not enough samples in one of the classes to plot the desired number of samples.")

    # Randomly select samples from each class
    random.seed(42)  # For reproducibility
    selected_real_indices = random.sample(real_indices, num_samples_per_class)
    selected_synthetic_indices = random.sample(synthetic_indices, num_samples_per_class)

    # Combine selected indices
    selected_indices = selected_real_indices + selected_synthetic_indices

    # Shuffle the selected indices for mixed class representation
    random.shuffle(selected_indices)

    # Create a matplotlib figure
    plt.figure(figsize=(15, 10))
    for idx, sample_idx in enumerate(selected_indices):
        img_path = all_image_paths[sample_idx]
        true_label = class_names[all_labels[sample_idx]]
        pred_label = class_names[all_preds[sample_idx]]

        try:
            # Open and process the image
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                # Convert to numpy array
                img_np = np.array(img)
                # Separate alpha channel
                alpha = img_np[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
                # Apply alpha channel to RGB channels
                img_np = img_np[:, :, :3]  # Get RGB channels
                img_np = img_np * np.expand_dims(alpha, axis=2)  # Apply alpha
                img_np = img_np.astype(np.uint8)  # Convert to uint8
                # Convert back to PIL Image
                img = Image.fromarray(img_np)
            else:
                # Ensure image is in RGB mode
                img = img.convert('RGB')
            
            # Apply the same transformations as during testing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Generate Grad-CAM heatmap (optional, if needed)
            # Assuming you have a Grad-CAM implementation, you can generate heatmaps here

            # For demonstration, we'll proceed without Grad-CAM

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Use a black image in case of error
            img = Image.new('RGB', (224, 224))
            input_tensor = torch.zeros((1, 3, 224, 224)).to(device)
            pred_label = "Error"

        # Plot the image
        ax = plt.subplot(num_samples_per_class * 2, 1, idx + 1)
        img_np = np.array(img) / 255.0  # Normalize to [0, 1]
        plt.imshow(img_np)
        plt.title(f'Predicted: {pred_label} | True: {true_label}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('balanced_sample_test_predictions_evaluation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()