import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from split import deterministic_split, random_split
from image_dataset import ImageDataset

# ===========================
# Set Random Seeds for Reproducibility
# ===========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ===========================
# Train Model Function
# ===========================
def train_model(datasets, variants, model_name, base_dir="/dtu/blackhole/12/145234/", num_epochs=10, batch_size=32, learning_rate=0.001):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    _datasets = []
    for dataset, variant in zip(datasets, variants):
        N_train_pets_real = 471
        N_train_cars_real = 2514
        n_train = N_train_cars_real if dataset=="cars" else N_train_pets_real


        target_dir = os.path.join(base_dir, f"{dataset}/{variant}")

        if dataset == "pets" and variant == "real-fewshot":
            target_dir = os.path.join(target_dir, "seed0")
            train_files, _ = deterministic_split(target_dir, test_ratio=0.2)
            dataset_obj = ImageDataset(file_list=train_files, transform=train_transform, synthetic_label=0)
            _datasets.append(dataset_obj)
        elif variant == "real-fewshot":
            target_dir = os.path.join(target_dir, "best")
            train_files, _ = deterministic_split(target_dir, test_ratio=0.2)
            dataset_obj = ImageDataset(file_list=train_files, transform=train_transform, synthetic_label=0)
            _datasets.append(dataset_obj)
        else:
            target_dir = os.path.join(target_dir, "train")
            train_files, _ = random_split(target_dir, n_train)
            dataset_obj = ImageDataset(file_list=train_files, transform=train_transform, synthetic_label=1)
            _datasets.append(dataset_obj)
        if not os.path.isdir(target_dir):
            raise ValueError(f"Target directory does not exist: {target_dir}")


    combined_dataset = ConcatDataset(_datasets)
    if len(combined_dataset) < 1000:
        num_epochs *= 3
    dataloader_train = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: "Real" and "Synthetic"
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(combined_dataset)
        epoch_acc = correct_predictions.double() / len(combined_dataset)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save the Model
    model_path = f"models/{model_name}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
