import os
import shutil
from pathlib import Path

# ===== CONFIG =====
IMAGES_DIR = "moana_xband_gray/images"      # grayscale images folder
LABELS_DIR = "moana_xband/labels"           # label folder (YOLO format)
OUTPUT_DIR = "moana_xband_gray_split"       # output base folder

CHUNK_SIZE = 20        # number of consecutive images per chunk
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}  # ratio by object count
# ==================

# Create output dirs
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Get sorted image list
images = sorted(Path(IMAGES_DIR).glob("*.png"))

# Create chunks
chunks = []
for i in range(0, len(images), CHUNK_SIZE):
    chunk_imgs = images[i:i+CHUNK_SIZE]
    chunk_labels = []
    total_objects = 0
    for img_path in chunk_imgs:
        label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
        chunk_labels.append(label_path)
        if label_path.exists():
            with open(label_path, "r") as f:
                total_objects += len(f.readlines())
    chunks.append({
        "images": chunk_imgs,
        "labels": chunk_labels,
        "objects": total_objects
    })

total_objects_all = sum(c["objects"] for c in chunks)
target_counts = {k: v * total_objects_all for k, v in SPLIT_RATIOS.items()}
current_counts = {k: 0 for k in SPLIT_RATIOS.keys()}

# Assign chunks to splits
splits_assignment = []
for chunk in chunks:
    # Pick split with lowest percentage towards target
    best_split = min(SPLIT_RATIOS.keys(),
                     key=lambda s: current_counts[s] / target_counts[s])
    splits_assignment.append((chunk, best_split))
    current_counts[best_split] += chunk["objects"]

# Copy files
for chunk, split in splits_assignment:
    for img_path, label_path in zip(chunk["images"], chunk["labels"]):
        shutil.copy2(img_path, f"{OUTPUT_DIR}/images/{split}/{img_path.name}")
        if label_path.exists():
            shutil.copy2(label_path, f"{OUTPUT_DIR}/labels/{split}/{label_path.name}")

# Summary
print("\nSplit Summary:")
for split in SPLIT_RATIOS.keys():
    img_count = len(list(Path(f"{OUTPUT_DIR}/images/{split}").glob("*.png")))
    obj_count = 0
    for label_file in Path(f"{OUTPUT_DIR}/labels/{split}").glob("*.txt"):
        with open(label_file, "r") as f:
            obj_count += len(f.readlines())
    print(f"{split.capitalize()}: {img_count} images, {obj_count} objects")
