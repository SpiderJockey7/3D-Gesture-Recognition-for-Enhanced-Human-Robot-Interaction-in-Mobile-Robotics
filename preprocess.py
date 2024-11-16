import os
from PIL import Image, ImageEnhance
import numpy as np
import random
import sys

# Parameters for processing
image_size = (224, 224)
rotation_range = (-45, 45)  # Random rotation range in degrees
brightness_range = (0.7, 1.3)  # Random brightness adjustment
contrast_range = (0.7, 1.3)  # Random contrast adjustment

# Function to apply data augmentation
def augment_image(img):
    # Random rotation
    angle = random.uniform(*rotation_range)
    img = img.rotate(angle)

    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    brightness_factor = random.uniform(*brightness_range)
    img = enhancer.enhance(brightness_factor)

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    contrast_factor = random.uniform(*contrast_range)
    img = enhancer.enhance(contrast_factor)

    return img

# Function to preprocess images
def preprocess_images(input_folder):
    # Define output folder in the same directory
    output_folder = os.path.join(os.path.dirname(input_folder), f"processed_{os.path.basename(input_folder)}")
    os.makedirs(output_folder, exist_ok=True)

    for label in os.listdir(input_folder):
        label_path = os.path.join(input_folder, label)
        if os.path.isdir(label_path):
            label_output_path = os.path.join(output_folder, label)
            os.makedirs(label_output_path, exist_ok=True)
            
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Open image
                        img = Image.open(img_path).convert('RGB')  # Keep color information for brightness/contrast
                        img = img.resize(image_size)  # Resize image
                        
                        # Apply data augmentation
                        img = augment_image(img)

                        # Normalize image (0-1 range)
                        img_array = np.array(img) / 255.0
                        processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
                        
                        # Save processed image to output folder
                        processed_img.save(os.path.join(label_output_path, img_file))
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    print(f"Processed images saved to: {output_folder}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 preprocess.py file_name")
        return

    # Get input folder from user
    input_path = sys.argv[1]

    # Check if the input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} does not exist!")
        return

    # Process the images
    preprocess_images(input_path)

if __name__ == "__main__":
    main()
