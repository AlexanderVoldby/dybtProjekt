import os
import shutil
import random
from math import floor

# Set a random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Define the paths to the original datasets
REAL_DIR = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/original folders/stanford-cars-real-train-fewshot'
SYNTHETIC_DIR = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/original folders/stanford-cars-synthetic-classwise-16'

# Define the output directory
OUTPUT_DIR = '/Users/fredmac/Documents/DTU-FredMac/Deep/dybtProjekt/data/stanford_cars/split_models'

# Define the split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def get_car_models(real_dir, synthetic_dir):
    """
    Retrieves the list of car model subfolders.
    Assumes both real and synthetic directories have the same subfolders.
    """
    real_models = set(os.listdir(real_dir))
    synthetic_models = set(os.listdir(synthetic_dir))
    
    # Ensure both directories have the same car models
    if real_models != synthetic_models:
        missing_in_synthetic = real_models - synthetic_models
        missing_in_real = synthetic_models - real_models
        error_msg = ""
        if missing_in_synthetic:
            error_msg += f"Models missing in synthetic directory: {missing_in_synthetic}\n"
        if missing_in_real:
            error_msg += f"Models missing in real directory: {missing_in_real}\n"
        raise ValueError(f"Mismatch in car models between real and synthetic directories.\n{error_msg}")
    
    return sorted(list(real_models))

def split_models(models, train_ratio, val_ratio, test_ratio):
    """
    Splits the list of models into train, val, and test sets based on the provided ratios.
    """
    total = len(models)
    train_count = floor(total * train_ratio)
    val_count = floor(total * val_ratio)
    test_count = total - train_count - val_count  # Ensure all models are assigned
    
    train_set = models[:train_count]
    val_set = models[train_count:train_count + val_count]
    test_set = models[train_count + val_count:]
    
    return train_set, val_set, test_set

def create_output_dirs(output_dir):
    """
    Creates the required output directories.
    """
    subsets = ['train', 'val', 'test']
    classes = ['real', 'synthetic']
    for cls in classes:
        for subset in subsets:
            dir_path = os.path.join(output_dir, cls, subset)
            os.makedirs(dir_path, exist_ok=True)

def copy_images(src_dir, dest_dir, model, subset):
    """
    Copies images from the source subfolder to the destination directory.
    Filenames will include the model name to prevent conflicts.
    """
    model_path = os.path.join(src_dir, model)
    if not os.path.isdir(model_path):
        print(f"Warning: {model_path} is not a directory. Skipping.")
        return
    
    for filename in os.listdir(model_path):
        src_file = os.path.join(model_path, filename)
        if os.path.isfile(src_file):
            # Create a new filename that includes the model name
            new_filename = f"{model}_{filename}"
            dest_file = os.path.join(dest_dir, new_filename)
            shutil.copy2(src_file, dest_file)

def main():
    print("Retrieving car models...")
    models = get_car_models(REAL_DIR, SYNTHETIC_DIR)
    print(f"Total car models found: {len(models)}")
    
    print("Shuffling and splitting car models into train, val, test sets...")
    random.shuffle(models)
    train_set, val_set, test_set = split_models(models, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    print(f"Number of models - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    print("Creating output directories...")
    create_output_dirs(OUTPUT_DIR)
    
    subsets = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    
    for subset, subset_models in subsets.items():
        print(f"\nProcessing subset: {subset} with {len(subset_models)} models")
        for cls, src_dir in [('real', REAL_DIR), ('synthetic', SYNTHETIC_DIR)]:
            dest_dir = os.path.join(OUTPUT_DIR, cls, subset)
            print(f"  Copying {cls} images to {dest_dir}")
            for model in subset_models:
                copy_images(src_dir, dest_dir, model, subset)
    
    print("\nData splitting completed successfully.")

if __name__ == "__main__":
    main()