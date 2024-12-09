import os
import hashlib
import random

def deterministic_split(data_dir, test_ratio=0.2):
    """
    Deterministically split images into train and test sets using hashing.

    Args:
        data_dir (str): Path to the folder containing subdirectories for classes.
        test_ratio (float): Proportion of the data to include in the test split.

    Returns:
        train_files (list): List of file paths for the training set.
        test_files (list): List of file paths for the test set.
    """
    train_files = []
    test_files = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, file_name)
                    # Compute a hash of the file name
                    file_hash = int(hashlib.sha256(file_name.encode()).hexdigest(), 16)
                    # Assign to train or test based on the hash value
                    if file_hash % 100 < test_ratio * 100:
                        test_files.append(file_path)
                    else:
                        train_files.append(file_path)

    print(f"Data dir: {data_dir}")
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of test files: {len(test_files)}")

    return train_files, test_files

def random_split(data_dir, train_count):
    """
    Randomly split images into train and test sets, specifying the number of training points.

    Args:
        data_dir (str): Path to the folder containing subdirectories for classes.
        train_count (int): Number of data points to include in the training set.

    Returns:
        train_files (list): List of file paths for the training set.
        test_files (list): List of file paths for the test set.
    """
    all_files = []

    # Gather all image files from all class directories
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(class_dir, file_name))
    
    # Shuffle the list to randomize the order
    random.shuffle(all_files)

    # Ensure there are enough files to meet the train count
    if len(all_files) < train_count:
        raise ValueError("Not enough data points to satisfy the specified number of training points.")

    # Split the files into training and testing sets
    train_files = all_files[:train_count]
    test_files = all_files[train_count:train_count*10]

    print(f"Dataset {data_dir}")
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of test files: {len(test_files)}")

    return train_files, test_files