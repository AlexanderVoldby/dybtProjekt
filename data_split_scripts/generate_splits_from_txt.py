import os
import shutil
import json
from pathlib import Path


# Kør scriptet to gange: En gang med "split_info_random.txt" og en gang med "split_info_models.txt"
# for hver gang, så navngiv output-mappen til hhv. "split_random" og "split_models"


SPLIT_INFO_FILE = "split_info_random.txt" # Kør en gang med "split__info_random.txt" og en gang med "split_info_models.txt"
OUTPUT_DIR = "/path/to/dybtProjekt/data/stanford_cars/split_random"  # lav "split_random"-mappe eller "split_models"-mappe svarende til hvilken split_info-fil der bruges
ORIGINAL_REAL_DIR = "/path/to/stanford-cars-real-train-fewshot"  
ORIGINAL_SYNTHETIC_DIR = "/path/to/stanford-cars-synthetic-classwise-16"  

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def load_split_info(split_info_file):
    with open(split_info_file, 'r') as f:
        split_info = json.load(f)
    print(f"Loaded split information from '{split_info_file}'.")
    return split_info

def copy_image(original_category_dir, class_name, image_name, destination_dir):
    source_image_path = original_category_dir / class_name / image_name
    if not source_image_path.exists():
        print(f"Error: Source image '{source_image_path}' does not exist.")
        return

    new_filename = f"{class_name}_{image_name}"
    destination_image_path = destination_dir / new_filename

    try:
        shutil.copy2(source_image_path, destination_image_path)
        print(f"Copied '{source_image_path}' to '{destination_image_path}'.")
    except Exception as e:
        print(f"Error copying '{source_image_path}' to '{destination_image_path}': {e}")

def replicate_split(split_info, original_real_dir, original_synthetic_dir, output_dir):
    categories = ['real', 'synthetic']
    subsets = ['train', 'val', 'test']
    
    for category in categories:
        for subset in subsets:
            filenames = split_info.get(category, {}).get(subset, [])
            if not filenames:
                print(f"Warning: No files found for Category: '{category}' | Subset: '{subset}'. Skipping...")
                continue
            
            destination_subset_dir = output_dir / category / subset
            destination_subset_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nProcessing Category: '{category}' | Subset: '{subset}' | Files: {len(filenames)}")
            
            for filename in filenames:
                if not filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    print(f"Skipping non-image file '{filename}'.")
                    continue
                
                if '_' not in filename:
                    print(f"Warning: Filename '{filename}' does not contain an underscore '_'. Skipping...")
                    continue
                
                class_name, original_image = filename.split('_', 1)
                if category == 'real':
                    original_category_dir = Path(original_real_dir)
                else:
                    original_category_dir = Path(original_synthetic_dir)
                
                copy_image(original_category_dir, class_name, original_image, destination_subset_dir)

def main():
    split_info_file = Path(SPLIT_INFO_FILE)
    original_real_dir = Path(ORIGINAL_REAL_DIR)
    original_synthetic_dir = Path(ORIGINAL_SYNTHETIC_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    if not split_info_file.exists() or not split_info_file.is_file():
        print(f"Error: split_info file '{split_info_file}' does not exist or is not a file.")
        return
    
    if not original_real_dir.exists() or not original_real_dir.is_dir():
        print(f"Error: Original real images directory '{original_real_dir}' does not exist or is not a directory.")
        return
    
    if not original_synthetic_dir.exists() or not original_synthetic_dir.is_dir():
        print(f"Error: Original synthetic images directory '{original_synthetic_dir}' does not exist or is not a directory.")
        return
    
    split_info = load_split_info(split_info_file)
    replicate_split(split_info, original_real_dir, original_synthetic_dir, output_dir)
    
    print("\nData replication completed successfully.")

if __name__ == "__main__":
    main()