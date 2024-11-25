import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def collect_image_paths(source_dir, image_extensions=None):
    """
    Recursively collects all image file paths from the source directory along with their class names.

    :param source_dir: Path to the source directory.
    :param image_extensions: Set of image file extensions to include.
    :return: List of tuples (image_file_path, class_name).
    """
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    source_path = Path(source_dir)
    image_info = []
    class_counts = defaultdict(int)
    
    # Iterate through all subdirectories
    for root, dirs, files in os.walk(source_path):
        # Determine relative path from source_dir
        relative_path = Path(root).relative_to(source_path)
        parts = relative_path.parts
        
        if len(parts) == 0:
            # Images directly under source_dir (if any)
            class_name = 'unknown'
        else:
            # Assume the first part is the class name
            class_name = parts[0]
        
        print(f"Scanning directory: {root} (Class: {class_name})")
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                image_info.append((file_path, class_name))
                class_counts[class_name] += 1
    
    # Log the number of images per class
    print("\nImage counts per class:")
    for cls, count in class_counts.items():
        print(f"  Class '{cls}': {count} images")
    
    return image_info

def split_data(image_info, train_ratio=0.8):
    """
    Splits the image info into training and testing sets.

    :param image_info: List of tuples (image_file_path, class_name).
    :param train_ratio: Proportion of images to include in the training set.
    :return: Tuple of (train_info, test_info).
    """
    random.shuffle(image_info)
    split_index = int(len(image_info) * train_ratio)
    train_info = image_info[:split_index]
    test_info = image_info[split_index:]
    return train_info, test_info

def copy_images(image_info, destination_dir):
    """
    Copies and renames images to the destination directory.

    :param image_info: List of tuples (image_file_path, class_name).
    :param destination_dir: Path to the destination directory.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    
    for img_path, class_name in image_info:
        try:
            # Create a new filename with class name prefix
            new_filename = f"{class_name}_{img_path.name}"
            destination_path = destination / new_filename
            
            # If the new filename already exists, append a unique identifier
            if destination_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                unique_id = random.randint(1000, 9999)
                new_filename = f"{class_name}_{stem}_{unique_id}{suffix}"
                destination_path = destination / new_filename
            
            shutil.copy2(img_path, destination_path)
        except Exception as e:
            print(f"Error copying {img_path} to {destination}: {e}")

def process_folder(source_folder, category_name, output_base, train_ratio=0.8):
    """
    Processes a single source folder: collects images, splits them, and copies to train/test with renamed files.

    :param source_folder: Path to the source outer folder.
    :param category_name: 'real' or 'synthetic'.
    :param output_base: Base directory where 'real' and 'synthetic' folders will be created.
    :param train_ratio: Proportion of images to include in the training set.
    """
    print(f"\nProcessing '{source_folder}' as category '{category_name}'...")
    image_info = collect_image_paths(source_folder)
    total_images = len(image_info)
    print(f"\nTotal images found in '{source_folder}': {total_images}")
    
    if total_images == 0:
        print(f"No images found in '{source_folder}'. Skipping...")
        return
    
    train_info, test_info = split_data(image_info, train_ratio)
    print(f"Split into {len(train_info)} training and {len(test_info)} testing images.")
    
    # Define destination directories
    train_dest = Path(output_base) / category_name / 'train'
    test_dest = Path(output_base) / category_name / 'test'
    
    # Copy and rename images
    print(f"Copying training images to '{train_dest}'...")
    copy_images(train_info, train_dest)
    
    print(f"Copying testing images to '{test_dest}'...")
    copy_images(test_info, test_dest)
    
    print(f"Finished processing category '{category_name}'.\n")

def main():
    # Define source folders
    real_source = "data/stanford-cars-real-train-fewshot"
    synthetic_source = "data/stanford-cars-synthetic-classwise-16"
    
    # Verify that source folders exist
    for folder in [real_source, synthetic_source]:
        if not Path(folder).exists():
            print(f"Source folder '{folder}' does not exist. Please check the path.")
            return
    
    # Define output base directory (current working directory)
    output_base = Path.cwd()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process 'real' category
    process_folder(real_source, 'real', output_base, train_ratio=0.8)
    
    # Process 'synthetic' category
    process_folder(synthetic_source, 'synthetic', output_base, train_ratio=0.8)
    
    print("All categories processed successfully.")

if __name__ == "__main__":
    main()