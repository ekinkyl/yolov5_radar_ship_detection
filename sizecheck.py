import os

from PIL import Image

# Path to your dataset
dataset_path = "moana_xband_gray_split_sequence_final/images/train"  # change to test/val if needed

# Loop through images and print size
for img_name in os.listdir(dataset_path):
    if img_name.endswith((".png", ".jpg", ".jpeg")):  # check common formats
        img_path = os.path.join(dataset_path, img_name)
        with Image.open(img_path) as img:
            print(f"{img_name}: {img.size}")  # (width, height)
