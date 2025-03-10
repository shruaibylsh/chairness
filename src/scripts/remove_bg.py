import cv2
import numpy as np
import os

def remove_background_grabcut(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create a mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the bounding box around the object (foreground)
    # You can adjust this bounding box to fit your chair
    height, width = image.shape[:2]
    rect = (50, 50, width - 100, height - 100)  # (x, y, w, h)

    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where the foreground is 1 (definite foreground) or 3 (probable foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    result = image * mask2[:, :, np.newaxis]

    # Create a white background
    white_background = np.ones_like(image, np.uint8) * 255

    # Combine the foreground and white background
    final_result = np.where(mask2[:, :, np.newaxis] == 1, result, white_background)

    # Save the result
    cv2.imwrite(output_path, final_result)

def process_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            remove_background_grabcut(input_path, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    # Define the paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    base_dir = os.path.dirname(script_dir)  # Move up to the "src" directory
    input_folder = os.path.join(base_dir, "data", "chair images")  # Path to the input folder
    output_folder = os.path.join(base_dir, "data", "chair images_no_bg")  # Path to the output folder
    
    # Process the folder
    process_folder(input_folder, output_folder)