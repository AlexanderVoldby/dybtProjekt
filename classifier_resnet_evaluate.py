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

def main():
    # ===========================
    # 3. Configuration and Hyperparameters
    # ===========================
    # Paths to the dataset directories
    data_dir = "data/stanford-cars/split_random/"  # Update this path
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
    # 4. Data Transforms
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
    # 5. Load Test Dataset and DataLoader
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
    # 6. Initialize the Model and Load Saved Weights
    # ===========================
    print("Initializing the model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the saved model weights
    model_path = 'models/resnet18_finetuned.pth'  # Update this path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # ===========================
    # 7. Define Loss Function
    # ===========================
    criterion = nn.CrossEntropyLoss()

    # ===========================
    # 8. Evaluation on Test Set
    # ===========================
    print("Starting evaluation on the test set...")
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_test, desc="Testing Phase"):
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

            # Collect all predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute average loss and accuracy
    test_loss = running_loss / dataset_size
    test_acc = running_corrects.float() / dataset_size

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

    # ===========================
    # 9. Confusion Matrix and Classification Report
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
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_set_evaluation.png', dpi=300)
    plt.show()

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)

    # ===========================
    # 10. (Optional) Visualize Sample Predictions
    # ===========================
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp * np.array(imagenet_std) + np.array(imagenet_mean), 0, 1)
        plt.imshow(inp)
        if title:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    print("Visualizing some test predictions...")
    # Get a batch of test data
    test_iter = iter(dataloader_test)
    inputs, classes = next(test_iter)
    inputs = inputs.to(device)
    classes = classes.to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    # Plot the first 6 images with their predictions
    plt.figure(figsize=(12, 8))
    for i in range(min(6, inputs.size(0))):
        ax = plt.subplot(2, 3, i+1)
        inp = inputs.cpu().data[i]
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp * np.array(imagenet_std) + np.array(imagenet_mean), 0, 1)
        plt.imshow(inp)
        ax.set_title(f'Predicted: {class_names[preds[i]]}\nTrue: {class_names[classes[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_test_predictions_evaluation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()