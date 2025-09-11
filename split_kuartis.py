import os
import shutil
import random

# Input folders
img_dir = "Kuartis_images"
label_dir = "Kuartis_labels"

# Output root folder
output_root = "Kuartis_split"

# Define ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

# Collect and sort images and labels
images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

# Double check: names must match
assert [os.path.splitext(f)[0] for f in images] == [os.path.splitext(f)[0] for f in labels], \
    "❌ Images and labels do not match!"

# Shuffle before splitting
combined = list(zip(images, labels))
random.shuffle(combined)

n = len(combined)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

splits = {
    "train": combined[:train_end],
    "val": combined[train_end:val_end],
    "test": combined[val_end:]
}

# Copy files
for split, files in splits.items():
    for img_file, label_file in files:
        # Copy image
        shutil.copy(os.path.join(img_dir, img_file),
                    os.path.join(output_root, "images", split, img_file))

        # Copy label
        shutil.copy(os.path.join(label_dir, label_file),
                    os.path.join(output_root, "labels", split, label_file))

print("✅ Split completed!")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
