import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Load the Hugging Face model
model_name = "briaai/RMBG-2.0"
model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
torch.set_float32_matmul_precision('high')  # Set matrix multiplication precision
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Image transformation settings
image_size = (1024, 1024)  # Input size expected by the model
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Comment in for real Stanford cars images / comment out for synthetic
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
end_folder_index = -1

# Get all subfolders in the input parent folder
all_folders = sorted(os.listdir(output_parent_folder))


# Limit the folders to process if specified
if not PROCESS_ALL_FOLDERS:
    all_folders = all_folders[-1]

# Iterate through selected subfolders
for folder in ["smart fortwo Convertible 2012"]:#all_folders:
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
                    # Load the input image
                    original_image = Image.open(input_image_path)

                    # Preprocess the image
                    input_images = transform_image(original_image).unsqueeze(0).to(device)

                    # Perform inference
                    with torch.no_grad():
                        preds = model(input_images)[-1].sigmoid().cpu()
                    
                    # Generate the alpha mask
                    pred = preds[0].squeeze()
                    pred_pil = transforms.ToPILImage()(pred)
                    mask = pred_pil.resize(original_image.size)

                    # Apply the mask to the original image
                    original_image.putalpha(mask)

                    # Determine the output file path
                    output_image_path = os.path.join(output_folder_path, filename)

                    # Handle image modes and formats
                    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                        # Convert to RGB for JPEGs
                        # Convert to RGB for JPEGs (removes alpha channel)
                        output_image_path = output_image_path.rsplit('.', 1)[0] + '.png'
                        original_image.save(output_image_path, format="PNG")
                    else:
                        original_image.save(output_image_path)

                    print(f"Processed: {folder}/{filename}")

                except Exception as e:
                    print(f"Error processing {folder}/{filename}: {e}")
            else:
                print(f"Skipping non-image file: {folder}/{filename}")
