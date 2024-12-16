import os
from rembg import remove
from PIL import Image

# Define parent folders

# Comment in for real stanford cars images / comment for synthetic
#input_parent_folder = 'data/stanford-cars-real-train-fewshot'
#output_parent_folder = 'data/no-background-stanford-cars-real-train-fewshot'

# Comment in for synthetic / comment out for real
input_parent_folder = 'data/stanford-cars-synthetic-classwise-16'
output_parent_folder = 'data/no-background-stanford-cars-synthetic-classwise-16'

os.makedirs(output_parent_folder, exist_ok=True)

# Specify which folders to process (e.g., the first 3 folders)
# Set to None to process all folders

PROCESS_ALL_FOLDERS = False

start_folder_index = 0
end_folder_index = 5


# Get all subfolders in the input parent folder
all_folders = sorted(os.listdir(input_parent_folder))

# Limit the folders to process if specified
if not PROCESS_ALL_FOLDERS:
    all_folders = all_folders[start_folder_index:end_folder_index]

# Iterate through selected subfolders
for folder in all_folders:
    input_folder_path = os.path.join(input_parent_folder, folder)

    # Check if the path is a folder
    if os.path.isdir(input_folder_path):
        # Create a corresponding output folder with whitespace removed
        clean_folder_name = folder.replace(' ', '')
        output_folder_path = os.path.join(output_parent_folder, clean_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        # Process images in the current folder
        for filename in os.listdir(input_folder_path):
            input_image_path = os.path.join(input_folder_path, filename)

            # Check if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Open the input image
                    input_image = Image.open(input_image_path)

                    # Remove the background
                    output_image = remove(input_image)

                    # Determine the output file path
                    output_image_path = os.path.join(output_folder_path, filename)

                    # Handle image modes and formats
                    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                        if output_image.mode == 'RGBA':
                            output_image = output_image.convert('RGB')
                        output_image.save(output_image_path, format="JPEG")
                    else:
                        output_image.save(output_image_path)

                    print(f"Processed: {folder}/{filename}")

                except Exception as e:
                    print(f"Error processing {folder}/{filename}: {e}")
            else:
                print(f"Skipping non-image file: {folder}/{filename}")
