import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm

from resnet18vis import make_resnet_dataset, get_resnet_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_images = 'data/stanford-cars-real-train-fewshot'
ai_generated_images = 'data/stanford-cars-synthetic-classwise-16'

# Define a dataset class for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings (numpy array): The extracted embeddings (features).
            labels (numpy array): Corresponding labels (0 for real, 1 for synthetic).
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Define a simple feedforward neural network classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

real_dataset = make_resnet_dataset(real_images, synthetic_label=0)
ai_dataset = make_resnet_dataset(ai_generated_images, synthetic_label=1) 

# Combine datasets
combined_dataset = real_dataset + ai_dataset
combined_data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

embeddings, synth_labels, class_labels, embeddings_tsne = get_resnet_embeddings(combined_data_loader)

# Prepare the dataset and DataLoader
embedding_dim = embeddings.shape[1]
dataset = EmbeddingDataset(embeddings, synth_labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = Classifier(input_dim=embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training classifier")
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} of {num_epochs}")
    model.train()
    train_loss = 0.0

    for embeddings_batch, labels_batch in train_loader:
        # Forward pass
        embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)
        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings_batch, labels_batch in val_loader:
            embeddings_batch, labels_batch = embeddings_batch.to(device), labels_batch.to(device)
            outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {accuracy:.4f}")

# Save the trained model
output_dir = './models'
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'synthetic_classifier.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
