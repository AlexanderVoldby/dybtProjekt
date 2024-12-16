import os
import random
import shutil
import sys

def collect_image_files(source_dir, image_extensions):
    """
    Traverse the source directory and collect all image file paths.
    
    :param source_dir: Path to the source directory.
    :param image_extensions: Tuple of acceptable image file extensions.
    :return: List of image file paths.
    """
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def select_random_images(image_files, number):
    """
    Select a specified number of random images from the list.
    
    :param image_files: List of image file paths.
    :param number: Number of images to select.
    :return: List of selected image file paths.
    """
    if len(image_files) < number:
        print(f"Warning: Only {len(image_files)} images found. All will be copied.")
        return image_files
    return random.sample(image_files, number)

def copy_images(selected_images, destination_dir):
    """
    Copy selected images to the destination directory.
    
    :param selected_images: List of image file paths to copy.
    :param destination_dir: Path to the destination directory.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")
    
    for idx, image_path in enumerate(selected_images, start=1):
        try:
            # Preserve the original filename
            filename = os.path.basename(image_path)
            destination_path = os.path.join(destination_dir, filename)
            
            # Handle potential filename conflicts
            if os.path.exists(destination_path):
                name, ext = os.path.splitext(filename)
                destination_path = os.path.join(destination_dir, f"{name}_{idx}{ext}")
            
            shutil.copy2(image_path, destination_path)
            print(f"Copied: {image_path} -> {destination_path}")
        except Exception as e:
            print(f"Failed to copy {image_path}. Error: {e}")

def main():
    """
    Main function to execute the script.
    """
    # You can modify these paths as needed
    source_dir = '/Users/fredmac/Documents/DTU-FredMac/sd2.1/train'
    destination_dir = '/Users/fredmac/Documents/DTU-FredMac/sd2.1/314/hello'
    number_of_images = 314  # Number of images to select

    # Validate source directory
    if not os.path.isdir(source_dir):
        print(f"Error: The source directory '{source_dir}' does not exist or is not a directory.")
        sys.exit(1)

    # Define acceptable image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    print("Collecting image files...")
    image_files = collect_image_files(source_dir, image_extensions)
    total_images = len(image_files)
    print(f"Total images found: {total_images}")

    if total_images == 0:
        print("No images found in the specified directory.")
        sys.exit(0)

    print(f"Selecting {number_of_images} random images...")
    selected_images = select_random_images(image_files, number_of_images)
    print(f"Number of images to be copied: {len(selected_images)}")

    print("Copying images...")
    copy_images(selected_images, destination_dir)
    print("Image copying completed.")

if __name__ == "__main__":
    main()