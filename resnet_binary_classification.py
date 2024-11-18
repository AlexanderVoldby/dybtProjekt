import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Define paths to the images
real_images = 'data/stanford-cars-real-train-fewshot'
ai_generated_images = 'data/stanford-cars-synthetic-classwise-16'

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, img_folder, label, transform=None):
        self.img_folder = img_folder
        self.label = label
        self.transform = transform
        self.images = []
        
        for img_file in os.listdir(img_folder):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(img_folder, img_file))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label

# Get common folders between real and AI-generated datasets
def get_common_folders(real_dir, synthetic_dir):
    real_folders = set(os.listdir(real_dir))
    synthetic_folders = set(os.listdir(synthetic_dir))
    return real_folders.intersection(synthetic_folders)

# Prepare datasets
common_folders = get_common_folders(real_images, ai_generated_images)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

real_dataset = []
ai_dataset = []

for folder in common_folders:
    real_path = os.path.join(real_images, folder)
    synthetic_path = os.path.join(ai_generated_images, folder)

    real_dataset.extend(ImageDataset(real_path, label=0, transform=transform))
    ai_dataset.extend(ImageDataset(synthetic_path, label=1, transform=transform))

combined_dataset = real_dataset + ai_dataset

# Split into training and testing datasets
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Sequential(
    nn.Linear(resnet18.fc.in_features, 1),  # Binary classification (1 output)
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(resnet18.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Testing and evaluation
resnet18.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = resnet18(images)
        predictions = (outputs > 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Convert to binary predictions and compute metrics
y_true = [int(label) for label in y_true]
y_pred = [int(pred[0]) for pred in y_pred]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}")
