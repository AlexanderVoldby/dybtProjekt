import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split  # For splitting datasets

class MagnitudePhaseSpectrumDataset(Dataset):
    """
    Custom Dataset for loading magnitude and phase spectra images from real and synthetic folders.
    Each sample consists of two channels: magnitude and phase.
    """
    def __init__(self, real_dir, synthetic_dir, transform=None):
        """
        Initializes the dataset by listing all image paths and assigning labels.

        Parameters:
            real_dir (str): Path to the real magnitude spectra images.
            synthetic_dir (str): Path to the synthetic magnitude spectra images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.real_dir = real_dir
        self.synthetic_dir = synthetic_dir
        self.transform = transform

        # List all magnitude image files in real and synthetic directories
        self.real_images = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'))]
        self.synthetic_images = [os.path.join(synthetic_dir, fname) for fname in os.listdir(synthetic_dir) 
                                 if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'))]

        # Labels: 0 for real, 1 for synthetic
        self.labels = [0]*len(self.real_images) + [1]*len(self.synthetic_images)
        self.image_paths = self.real_images + self.synthetic_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the magnitude and phase images and their label at the specified index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): Stacked magnitude and phase image tensor with 2 channels.
            label (int): Label of the image (0: real, 1: synthetic).
        """
        magnitude_path = self.image_paths[idx]
        label = self.labels[idx]

        # Derive the phase image path by replacing 'magnitude' with 'phase' in the path
        phase_path = magnitude_path.replace('magnitude', 'phase')

        try:
            # Open magnitude and phase images
            magnitude_image = Image.open(magnitude_path).convert('L')  # Convert to grayscale
            phase_image = Image.open(phase_path).convert('L')          # Convert to grayscale

            # Apply transforms if provided
            if self.transform:
                magnitude_image = self.transform(magnitude_image)
                phase_image = self.transform(phase_image)

            # Ensure both images have the correct shape
            if magnitude_image.shape != (1, 224, 224) or phase_image.shape != (1, 224, 224):
                raise ValueError(f"Transformed image shapes do not match expected [1, 224, 224]. "
                                 f"Magnitude: {magnitude_image.shape}, Phase: {phase_image.shape}")

            # Stack magnitude and phase images to create a 2-channel image
            image = torch.cat([magnitude_image, phase_image], dim=0)  # Shape: [2, 224, W]
        except Exception as e:
            print(f"Error loading images {magnitude_path} and/or {phase_path}: {e}")
            # Return a zero tensor with 2 channels and label -1 to indicate an error
            image = torch.zeros(2, 224, 224)
            label = -1

        return image, label

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    """
    Trains the model and returns the best model based on validation accuracy.

    Parameters:
        model (nn.Module): The neural network model to train.
        dataloaders (dict): Dictionary containing 'train', 'val', and 'test' DataLoaders.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int): Number of training epochs.

    Returns:
        model (nn.Module): The best-performing model.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            total_samples = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                # Filter out any samples with label -1 (errors)
                mask = labels != -1
                inputs = inputs[mask]
                labels = labels[mask]

                if inputs.numel() == 0:
                    continue  # Skip if no valid samples in the batch

                inputs = inputs.to(device).float()  # Ensure float32
                labels = labels.to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize only if in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / total_samples if total_samples > 0 else 0
            epoch_acc = running_corrects.float() / total_samples if total_samples > 0 else 0

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the test set and prints performance metrics.

    Parameters:
        model (nn.Module): The trained neural network model.
        dataloader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computation on (CPU or GPU).
    """
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing Phase"):
            # Filter out any samples with label -1 (errors)
            mask = labels != -1
            inputs = inputs[mask]
            labels = labels[mask]

            if inputs.numel() == 0:
                continue  # Skip if no valid samples in the batch

            inputs = inputs.to(device).float()  # Ensure float32
            labels = labels.to(device).long()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    test_loss = running_loss / total_samples if total_samples > 0 else 0
    test_acc = running_corrects.float() / total_samples if total_samples > 0 else 0

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

def main():
    # Paths to the magnitude spectra folders
    # Update these paths based on your directory structure
    output_root = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/spectra_output/'

    # Define train and test directories for magnitude
    train_real_dir = os.path.join(output_root, 'train', 'real', 'magnitude')
    train_synthetic_dir = os.path.join(output_root, 'train', 'synthetic', 'magnitude')
    test_real_dir = os.path.join(output_root, 'test', 'real', 'magnitude')
    test_synthetic_dir = os.path.join(output_root, 'test', 'synthetic', 'magnitude')

    # Check if directories exist
    for dir_path in [train_real_dir, train_synthetic_dir, test_real_dir, test_synthetic_dir]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")

    # Define transformations without converting to 3 channels
    # Use single-channel normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize for single channel
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize for single channel
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize for single channel
        ]),
    }

    # Create the full training dataset
    full_train_dataset = MagnitudePhaseSpectrumDataset(real_dir=train_real_dir, synthetic_dir=train_synthetic_dir, transform=data_transforms['train'])

    # Determine the size of the full training dataset
    total_train_size = len(full_train_dataset)

    # Calculate sizes for train and validation splits
    train_size = int(0.875 * total_train_size)  # 87.5% of train = 70% overall
    val_size = total_train_size - train_size    # 12.5% of train = 10% overall

    # Split the full training dataset into train and validation sets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Assign validation transforms (if different from training)
    val_dataset.dataset.transform = data_transforms['val']

    # Create the test dataset
    test_dataset = MagnitudePhaseSpectrumDataset(real_dir=test_real_dir, synthetic_dir=test_synthetic_dir, transform=data_transforms['test'])

    # Create DataLoaders
    batch_size = 32
    num_workers = 0  # Set to 0 to avoid multiprocessing issues on macOS

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained ResNet18
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        # Fallback if specific weights not found
        model = models.resnet18(pretrained=True)

    # Modify the first convolution layer to accept 2 channels instead of 3
    original_conv = model.conv1
    new_conv = nn.Conv2d(
        2, 
        original_conv.out_channels, 
        kernel_size=original_conv.kernel_size, 
        stride=original_conv.stride, 
        padding=original_conv.padding, 
        bias=(original_conv.bias is not None)
    )

    # Initialize the new_conv weights by copying the weights of the first two channels from the original conv
    with torch.no_grad():
        if original_conv.weight.shape[1] == 3:
            # Average the weights of the original three channels to initialize the two new channels
            new_conv.weight[:, 0, :, :] = original_conv.weight[:, 0, :, :]
            new_conv.weight[:, 1, :, :] = original_conv.weight[:, 1, :, :]
        else:
            # If original conv does not have 3 input channels, initialize randomly or as needed
            new_conv.weight[:, 0, :, :].copy_(original_conv.weight[:, 0, :, :])
            new_conv.weight[:, 1, :, :].copy_(original_conv.weight[:, 1, :, :])

    model.conv1 = new_conv

    # Modify the final layer to have 2 output classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Move model to the appropriate device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Only parameters of final layer and first conv layer are being optimized
    # You can choose to fine-tune more layers if needed
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Optionally, you can use a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 10
    best_model = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=num_epochs)

    # Save the fine-tuned model
    save_dir = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/models/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'resnet18_finetuned.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

    # Evaluate the model on the test set
    print("Evaluating the best model on the test set...")
    evaluate_model(best_model, dataloaders['test'], criterion, device)

if __name__ == "__main__":
    # main()


    def count_image_files(folder_path):
        total_files = 0
        valid_extensions = (".png", ".jpg")
        
        for root, _, files in os.walk(folder_path):
            total_files += sum(1 for file in files if file.lower().endswith(valid_extensions))
        
        return total_files

    # Specify your folder path
    folder_path = "/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-synthetic-classwise-16"
    folder_path = "/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford-cars-real-train-fewshot"
    print(f"Total number of image files: {count_image_files(folder_path)}")