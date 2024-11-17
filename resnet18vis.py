import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
import os

# Define paths to the images
real_images = 'data/stanford-cars-real-train-fewshot'
ai_generated_images = 'data/stanford-cars-synthetic-classwise-16'

# Custom Dataset Class with synthetic/real labels and class labels
class ImageDataset(Dataset):
    def __init__(self, img_folder, synthetic_label, transform=None):
        self.img_folder = img_folder
        self.synthetic_label = synthetic_label
        self.transform = transform
        self.images = []
        
        # Collect images and their class labels based on subdirectory names
        for class_folder in os.listdir(img_folder):
            class_path = os.path.join(img_folder, class_folder)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append((os.path.join(class_path, img_file), class_folder))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, class_label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert class label to an integer or a unique identifier (if classes are strings, use a mapping)
        return image, class_label, self.synthetic_label

def make_resnet_dataset(images_path, synthetic_label):
    # Define transformations and load the datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return ImageDataset(images_path, synthetic_label, transform=transform)

# Load datasets with synthetic label as 0 for real, 1 for AI-generated
real_dataset = make_resnet_dataset(real_images, synthetic_label=0)
print(len(real_dataset))
ai_dataset = make_resnet_dataset(ai_generated_images, synthetic_label=1)
print(len(ai_dataset))

# Combine datasets
combined_dataset = real_dataset + ai_dataset
combined_data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

def get_resnet_embeddings(data_loader, model_path="./cifar100_pretrained_resnet18.pth",
                            num_components_tsne=2):
    
    # Load pretrained ResNet18 model on CIFAR-100
    resnet18 = models.resnet18(pretrained=False)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)  # Adjust the final layer for CIFAR-100

    # Load the saved model weights
    resnet18.load_state_dict(torch.load(model_path))
    resnet18.eval()  # Set to evaluation mode for feature extraction

    # Remove the final classification layer
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])

    # Mapping class labels (e.g., "car1") to numeric labels for visualization
    unique_classes = list(set([label for _, label, _ in combined_dataset]))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    # Extract embeddings
    embeddings = []
    synthetic_labels = []
    class_labels = []

    with torch.no_grad():
        for images, class_label, synthetic_label in data_loader:
            features = resnet18(images).squeeze()
            embeddings.append(features)
            
            # Convert class label to numeric using the mapping
            numeric_class_labels = [class_to_idx[cl] for cl in class_label]
            class_labels.extend(numeric_class_labels)
            
            synthetic_labels.extend(synthetic_label.tolist())

    # Convert to numpy arrays
    embeddings = torch.cat(embeddings).numpy()
    synthetic_labels = np.array(synthetic_labels)
    print(np.sum(synthetic_labels))
    class_labels = np.array(class_labels)

    # Use T-SNE to visualize the embeddings
    tsne = TSNE(n_components=num_components_tsne, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    return embeddings, synthetic_labels, class_labels, embeddings_tsne

embeddings, synth_labels, class_labels, embeddings_tsne = get_resnet_embeddings(combined_data_loader)

# Create output directory if it doesn't exist
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Plot the T-SNE results with synthetic/real labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=synth_labels, cmap='coolwarm', alpha=0.7)
handles, _ = scatter.legend_elements()
plt.legend(handles, ['Real', 'AI-generated'], title="Label")
plt.title("T-SNE Visualization of Image Embeddings (Synthetic vs. Real)")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
# Save the figure
plt.savefig(f"{output_dir}/tsne_synthetic_vs_real.png")
plt.close()  # Close the plot to free memory

# Plot the T-SNE results color-coded by class labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=class_labels, cmap='viridis', alpha=0.7)
plt.title("T-SNE Visualization of Image Embeddings by Class")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
# Save the figure
plt.savefig(f"{output_dir}/tsne_by_class.png")
plt.close()  # Close the plot to free memory


