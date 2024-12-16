import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from skimage import io
from transformers import AutoModelForImageSegmentation
import torch
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModelForImageSegmentation

def remove_background(input_image: Image.Image) -> Image.Image:
    """
    Removes the background from the input image using a pre-trained segmentation model.

    Args:
        input_image (PIL.Image.Image): The input image from which to remove the background.

    Returns:
        PIL.Image.Image: The image with the background removed (transparent background).
    """
    # Load the pre-trained model
    model_name = "briaai/RMBG-1.4"
    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    
    # Set device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert PIL Image to NumPy array
    im = np.array(input_image)
    im_size = im.shape[0:2]  # Original image size

    # Preprocessing
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW
    model_input_size = [512, 512]  # Model's expected input size
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)  # Normalize to [0, 1]
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])  # Further normalization
    image = image.to(device)

    # Inference
    with torch.no_grad():
        result = model(image)

    # Postprocessing
    result = torch.squeeze(F.interpolate(result[0][0], size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)  # Normalize to [0, 1]
    im_array = (result * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)

    # Create mask and apply to original image
    mask = Image.fromarray(im_array).convert("L")  # Convert to grayscale mask
    no_bg_image = Image.new("RGBA", input_image.size, (0, 0, 0, 0))  # Transparent background
    input_image = input_image.convert("RGBA")  # Ensure input has alpha channel
    no_bg_image.paste(input_image, mask=mask)

    return no_bg_image

# Preprocessing function
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

# Postprocessing function
def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array



if __name__ == '__main__':
    # Load the Hugging Face model
    model_name = "briaai/RMBG-1.4"
    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)

    # Ensure device usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Define parent folders

    # Comment in for real Stanford cars images / comment out for synthetic
    # input_parent_folder = 'data/stanford-cars-real-train-fewshot'
    # output_parent_folder = 'data/no-background-stanford-cars-real-train-fewshot'
    # Comment in for synthetic / comment out for real
    input_parent_folder = 'data/stanford-cars-synthetic-classwise-16'
    output_parent_folder = 'data/v2-no-background-stanford-cars-synthetic-classwise-16'

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
                        # Load the input image
                        orig_im = io.imread(input_image_path)
                        orig_im_size = orig_im.shape[0:2]

                        # Preprocess the image
                        model_input_size = [512, 512]  # Adjust to match the model's expected size
                        image = preprocess_image(orig_im, model_input_size).to(device)

                        # Run inference
                        result = model(image)

                        # Post-process the result
                        result_image = postprocess_image(result[0][0], orig_im_size)

                        # Create the no-background image
                        pil_im = Image.fromarray(result_image)
                        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                        orig_image = Image.open(input_image_path)
                        no_bg_image.paste(orig_image, mask=pil_im)

                        # Determine the output file path
                        output_image_path = os.path.join(output_folder_path, filename)

                        # Handle image modes and formats
                        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                            no_bg_image = no_bg_image.convert('RGB')
                            no_bg_image.save(output_image_path, format="JPEG")
                        else:
                            no_bg_image.save(output_image_path)

                        print(f"Processed: {folder}/{filename}")

                    except Exception as e:
                        print(f"Error processing {folder}/{filename}: {e}")
                else:
                    print(f"Skipping non-image file: {folder}/{filename}")
