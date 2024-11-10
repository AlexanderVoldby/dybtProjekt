import torch
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
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

# Define transformations and load the datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected input size for CLIP
    transforms.ToTensor()           # Convert to tensor with values in the [0, 1] range
])

# Load datasets with synthetic label as 0 for real, 1 for AI-generated
real_dataset = ImageDataset(real_images, synthetic_label=0, transform=transform)
ai_dataset = ImageDataset(ai_generated_images, synthetic_label=1, transform=transform)

# Combine datasets
combined_dataset = real_dataset + ai_dataset
data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # Set to evaluation mode for feature extraction
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Mapping class labels (e.g., "car1") to numeric labels for visualization
unique_classes = list(set([label for _, label, _ in combined_dataset]))
class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}

# Extract embeddings
embeddings = []
synthetic_labels = []
class_labels = []

with torch.no_grad():
    for images, class_label, synthetic_label in data_loader:
        # Use the processor to prepare the images
        inputs = processor(images=images, return_tensors="pt", do_rescale=False)
        
        # Pass the images through the model and get image embeddings
        outputs = model.get_image_features(**inputs).squeeze()
        embeddings.append(outputs)
        
        # Convert class label to numeric using the mapping
        numeric_class_labels = [class_to_idx[cl] for cl in class_label]
        class_labels.extend(numeric_class_labels)
        
        synthetic_labels.extend(synthetic_label.tolist())

# Convert to numpy arrays
embeddings = torch.cat(embeddings).numpy()
synthetic_labels = np.array(synthetic_labels)
class_labels = np.array(class_labels)

# Use T-SNE to visualize the embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# Create output directory if it doesn't exist
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Plot the T-SNE results with synthetic/real labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=synthetic_labels, cmap='coolwarm', alpha=0.7)
handles, _ = scatter.legend_elements()
plt.legend(handles, ['Real', 'AI-generated'], title="Label")
plt.title("T-SNE Visualization of Image Embeddings (Synthetic vs. Real)")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
# Save the figure
plt.savefig(f"{output_dir}/CLIP_tsne_synthetic_vs_real.png")
plt.close()  # Close the plot to free memory

# Plot the T-SNE results color-coded by class labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=class_labels, cmap='viridis', alpha=0.7)
plt.title("T-SNE Visualization of Image Embeddings by Class")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
# Save the figure
plt.savefig(f"{output_dir}/CLIP_tsne_by_class.png")
plt.close()  # Close the plot to free memory

