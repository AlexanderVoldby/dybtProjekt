import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Transformations for CIFAR-100
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),  # CIFAR-100 normalization
])

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load ResNet18 model
model = models.resnet18(pretrained=False)  # Initialize with random weights
model.fc = nn.Linear(model.fc.in_features, 100)  # Update final layer for 100 classes (CIFAR-100)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Starting training on CIFAR-100...")

for epoch in range(num_epochs):
    print(f"epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Training complete.")

# Save the model
model_path = './cifar100_pretrained_resnet18.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")

