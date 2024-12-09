import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
from io import BytesIO

data_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/no-background/split_random/'

save_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/'  # Update this path if necessary
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'resnet18_finetuned_random_crop40_noback.pth')


real_train_dir = os.path.join(data_dir, 'real', 'train')
synthetic_train_dir = os.path.join(data_dir, 'synthetic', 'train')
real_val_dir = os.path.join(data_dir, 'real', 'val')
synthetic_val_dir = os.path.join(data_dir, 'synthetic', 'val')
real_test_dir = os.path.join(data_dir, 'real', 'test')
synthetic_test_dir = os.path.join(data_dir, 'synthetic', 'test')

# Verify that all directories exist
for dir_path in [real_train_dir, synthetic_train_dir, real_val_dir, synthetic_val_dir, real_test_dir, synthetic_test_dir]:
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")


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
# 2. Define Custom Transform Class
# ===========================
class RandomMaskedCenterCrop:
    """
    Custom transform that performs a random 40x40 crop centered on a randomly selected
    pixel within the masked (non-background) region of the image. The cropped region is
    then resized to 224x224 pixels.
    """
    def __init__(self, crop_size=40, resize_size=224):
        """
        Initializes the transform.

        Parameters:
            crop_size (int): The size of the square crop (default: 40).
            resize_size (int): The size to resize the cropped image to (default: 224).
        """
        self.crop_size = crop_size
        self.resize_size = resize_size

    def __call__(self, img):
        """
        Applies the transform to the given image.

        Parameters:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        # Check if the image has an alpha channel
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Extract the alpha channel as a mask
            alpha = img.split()[-1]
            alpha_np = np.array(alpha)

            # Find coordinates of all non-background pixels (alpha > 0)
            masked_pixels = np.argwhere(alpha_np > 0)

            if len(masked_pixels) == 0:
                # No masked pixels found; perform a standard random crop
                img_cropped = transforms.RandomCrop(self.crop_size)(img)
            else:
                # Randomly select one masked pixel as the center
                center_y, center_x = masked_pixels[random.randint(0, len(masked_pixels) - 1)]

                # Define the crop boundaries
                left = center_x - self.crop_size // 2
                upper = center_y - self.crop_size // 2

                # Ensure the crop is within image boundaries
                left = max(left, 0)
                upper = max(upper, 0)
                right = left + self.crop_size
                lower = upper + self.crop_size

                # Adjust if the crop goes beyond the image size
                img_width, img_height = img.size
                if right > img_width:
                    right = img_width
                    left = right - self.crop_size
                if lower > img_height:
                    lower = img_height
                    upper = lower - self.crop_size

                # Perform the crop
                img_cropped = img.crop((left, upper, right, lower))

            # Composite the cropped image onto a white background if it has an alpha channel
            if img_cropped.mode in ('RGBA', 'LA') or (img_cropped.mode == 'P' and 'transparency' in img_cropped.info):
                background = Image.new('RGB', img_cropped.size, (255, 255, 255))
                background.paste(img_cropped, mask=img_cropped.split()[-1])  # Use the alpha channel as mask
                img_cropped = background
            else:
                img_cropped = img_cropped.convert('RGB')

            # Resize the cropped image to the desired size
            img_resized = transforms.Resize(self.resize_size)(img_cropped)
            return img_resized

        else:
            # If there's no alpha channel, perform a standard random crop and resize
            img_cropped = transforms.RandomCrop(self.crop_size)(img)
            img_resized = transforms.Resize(self.resize_size)(img_cropped)
            return img_resized

# ===========================
# 3. Define Custom Dataset Class
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
        Keeps the alpha channel if present for the custom transform.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Label of the image (0: real, 1: synthetic).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Determine the file extension
            _, ext = os.path.splitext(img_path)
            ext = ext.lower()

            # Open image
            image = Image.open(img_path)

            # Keep the alpha channel for the custom transform
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image and label -1 to indicate an error
            image = Image.new('RGB', (224, 224))
            label = -1

        if self.transform:
            image = self.transform(image)

        return image, label

# ===========================
# 4. Main Function
# ===========================
def main():
    # ===========================
    # 4.1 Configuration and Hyperparameters
    # ===========================
    # Paths to the dataset directories
    data_dir = "data/stanford-cars/split_random/"
    real_train_dir = os.path.join(data_dir, 'real', 'train')
    synthetic_train_dir = os.path.join(data_dir, 'synthetic', 'train')
    real_val_dir = os.path.join(data_dir, 'real', 'val')
    synthetic_val_dir = os.path.join(data_dir, 'synthetic', 'val')
    real_test_dir = os.path.join(data_dir, 'real', 'test')
    synthetic_test_dir = os.path.join(data_dir, 'synthetic', 'test')





    # Hyperparameters
    num_epochs = 1
    batch_size = 32
    learning_rate = 1e-4
    num_classes = 2  # 'real' and 'synthetic'

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===========================
    # 4.2 Data Transforms
    # ===========================
    # ImageNet statistics for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Define transforms for training, validation, and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            RandomMaskedCenterCrop(crop_size=40, resize_size=224),  # Custom crop based on masked region
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        'val': transforms.Compose([
            RandomMaskedCenterCrop(crop_size=40, resize_size=224),    # Custom crop for validation
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        'test': transforms.Compose([
            RandomMaskedCenterCrop(crop_size=40, resize_size=224),    # Custom crop for testing
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
    }

    # ===========================
    # 4.3 Load Datasets
    # ===========================
    print("Loading datasets...")
    # Create the training dataset
    train_dataset = ImageDataset(real_dir=real_train_dir, 
                                 synthetic_dir=synthetic_train_dir, 
                                 transform=data_transforms['train'])

    # Create the validation dataset
    val_dataset = ImageDataset(real_dir=real_val_dir, 
                               synthetic_dir=synthetic_val_dir, 
                               transform=data_transforms['val'])

    # Create the test dataset
    test_dataset = ImageDataset(real_dir=real_test_dir, 
                                synthetic_dir=synthetic_test_dir, 
                                transform=data_transforms['test'])

    # ===========================
    # 4.4 Create DataLoaders
    # ===========================
    print("Creating DataLoaders...")
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    # Define class names
    class_names = ['real', 'synthetic']
    print(f"Classes: {class_names}")

    # ===========================
    # 4.5 Initialize the Model
    # ===========================
    print("Initializing the model...")
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # ===========================
    # 4.6 Define Loss Function and Optimizer
    # ===========================
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optionally, define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ===========================
    # 4.7 Training the Model
    # ===========================
    print("Starting training...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track gradients only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # ===========================
    # 4.8 Plot Training and Validation Metrics
    # ===========================
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png', dpi=300)
    plt.show()

    # ===========================
    # 4.9 Save the Best Model
    # ===========================
    save_dir = 'models/'  # Update this path if necessary
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'resnet18_finetuned.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # ===========================
    # 4.10 Evaluate the Model on the Test Set
    # ===========================
    print("Evaluating the model on the test set...")
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc="Testing Phase"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.float() / dataset_sizes['test']

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # ===========================
    # 4.11 Confusion Matrix and Classification Report
    # ===========================
    print("Generating confusion matrix and classification report...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix on Test Set')
    plt.savefig('confusion_matrix_test_set.png', dpi=300)
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ===========================
    # 4.12 Plot Sample Predictions (Optional)
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
    inputs, classes_batch = next(iter(dataloaders['test']))
    inputs = inputs.to(device)
    classes_batch = classes_batch.to(device)

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
        # Denormalize the image
        img = inp.numpy().transpose((1, 2, 0))
        img = img * np.array(imagenet_std) + np.array(imagenet_mean)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        ax.set_title(f'Predicted: {class_names[preds[i]]}\nTrue: {class_names[classes_batch[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_test_predictions.png', dpi=300)
    plt.show()

    # ===========================
    # 4.13 Plot Sample Images from Both Classes
    # ===========================
    print("Plotting sample images from both classes...")
    samples_per_class = 5
    sample_indices = {0: [], 1: []}

    # Collect sample indices
    for idx, label in enumerate(train_dataset.labels):
        if len(sample_indices[label]) < samples_per_class:
            sample_indices[label].append(idx)
        if all(len(indices) == samples_per_class for indices in sample_indices.values()):
            break

    plt.figure(figsize=(12, 6))
    for class_label, indices in sample_indices.items():
        for i, idx in enumerate(indices):
            ax = plt.subplot(2, samples_per_class, i + 1 + class_label * samples_per_class)
            image, label = train_dataset[idx]
            # Denormalize the image
            img = image.numpy().transpose((1, 2, 0))
            img = img * np.array(imagenet_std) + np.array(imagenet_mean)
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            ax.set_title(f'Class: {class_names[label]}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_class_images.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()