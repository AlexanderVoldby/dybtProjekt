import os
import shutil
from pathlib import Path
from random import sample

# Base paths for datasets
base_path = "/dtu/blackhole/12/145234/"
output_base_path = "/work3/s214591/dybtProjekt/small_dataset"
datasets = ["pets", "cars"]

# Pre-specified classes for each dataset
preselected_classes = {
    "pets": ["Samoyed", "Persian", "Scottish Terrier"],  # Replace with actual class folder names
    "cars": ["Audi R8 Coupe 2012", "BMW X3 SUV 2012", "Rolls-Royce Phantom Sedan 2012"]  # Replace with actual class folder names
}

# Number of images to copy per class
images_per_class = 10

def copy_images(dataset_name, source_folder, target_folder, class_name):
    # Create target class folder if it doesn't exist
    target_class_path = os.path.join(target_folder, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    # Source class folder path
    source_class_path = os.path.join(source_folder, class_name)

    # List all image files in the class folder
    all_images = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]

    # Select 10 random images from the folder
    selected_images = sample(all_images, min(images_per_class, len(all_images)))

    # Copy selected images to the target class folder
    for image in selected_images:
        source_image_path = os.path.join(source_class_path, image)
        target_image_path = os.path.join(target_class_path, image)
        shutil.copyfile(source_image_path, target_image_path)

    print(f"Copied {len(selected_images)} images from class '{class_name}' in dataset '{dataset_name}'.")



for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    output_dataset_path = os.path.join(output_base_path, dataset)

    # Iterate through 3 subfolders in the dataset
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        print(subfolder_path)
        for _subfolder in os.listdir(subfolder_path):
            _subfolder_path = os.path.join(subfolder_path, _subfolder)
            print(_subfolder_path)

            if os.path.isdir(_subfolder_path):
                output_subfolder_path = os.path.join(output_dataset_path, subfolder, _subfolder)
                print(output_subfolder_path)

                # Iterate through the pre-specified classes for the current dataset
                for class_name in preselected_classes[dataset]:
                    class_path = os.path.join(_subfolder_path, class_name)

                    if os.path.isdir(class_path):
                        copy_images(dataset, _subfolder_path, output_subfolder_path, class_name)
