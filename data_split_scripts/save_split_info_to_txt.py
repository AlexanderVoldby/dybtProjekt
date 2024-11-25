import os
import json
from pathlib import Path

# Define paths using variables
SPLIT_MODELS_DIR = "/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/split_random"  # Replace with the actual path to your split_models directory
SPLIT_INFO_FILE = "/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/split_info_random.txt"  # Replace with the desired path for the split_info.txt file

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def extract_split_info(split_models_dir):
    split_info = {}
    categories = ['real', 'synthetic']
    subsets = ['train', 'val', 'test']
    
    for category in categories:
        split_info[category] = {}
        for subset in subsets:
            subset_dir = split_models_dir / category / subset
            if not subset_dir.exists() or not subset_dir.is_dir():
                print(f"Warning: Directory '{subset_dir}' does not exist. Skipping...")
                split_info[category][subset] = []
                continue
            filenames = [
                filename for filename in os.listdir(subset_dir)
                if filename.lower().endswith(tuple(IMAGE_EXTENSIONS))
            ]
            split_info[category][subset] = filenames
            print(f"Category: '{category}' | Subset: '{subset}' | Files: {len(filenames)}")
    return split_info

def save_split_info(split_info, split_info_file):
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=4)
    print(f"\nSplit information saved to '{split_info_file}'.")

def main():
    split_models_dir = Path(SPLIT_MODELS_DIR)
    split_info_file = Path(SPLIT_INFO_FILE)
    
    if not split_models_dir.exists() or not split_models_dir.is_dir():
        print(f"Error: split_models directory '{split_models_dir}' does not exist or is not a directory.")
        return
    
    split_info = extract_split_info(split_models_dir)
    save_split_info(split_info, split_info_file)

if __name__ == "__main__":
    main()