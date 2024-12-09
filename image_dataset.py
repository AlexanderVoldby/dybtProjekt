import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, file_list, transform=None, synthetic_label=None):
        """
        Args:
            image_files (list): List of file paths for images.
            labels (list, optional): List of labels corresponding to the image files.
                                     If None, labels will be inferred based on folder names.
            transform (callable, optional): Transform to apply to the images.
            synthetic_label (int, optional): Label to assign all images if labels are not provided.
        """
        self.image_paths = file_list
        self.transform = transform
        self.synthetic_label = synthetic_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.synthetic_label

        try:
            # Load the image and convert to RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            label = -1

        if self.transform:
            image = self.transform(image)

        # Use the synthetic_label if provided, otherwise use the inferred label
        return image, label
