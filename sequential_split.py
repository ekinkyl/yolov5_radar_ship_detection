import os
import shutil
from pathlib import Path
from math import floor

# ===== CONFIG =====
IMAGES_DIR = "moana_xband_gray/images"      # grayscale images folder
LABELS_DIR = "moana_xband/labels"           # label folder (YOLO format)
OUTPUT_DIR = "moana_xband_gray_split_sequence_final"       # output base folder

SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
GROUP_SIZE = 3  # keep 3 consecutive frames together
# ==================

# Create output dirs
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Get sorted image list
images = sorted(Path(IMAGES_DIR).glob("*.png"))
n = len(images)
if n == 0:
    raise RuntimeError(f"No .png images found in {IMAGES_DIR}")

# Group images into triplets (last 1–2 images kept as remainder)
full_group_count = n // GROUP_SIZE
remainder_imgs = images[full_group_count * GROUP_SIZE:]  # 0,1, or 2 images
groups = [images[i*GROUP_SIZE:(i+1)*GROUP_SIZE] for i in range(full_group_count)]

# Compute how many full groups go to each split (train/val by floor, test gets the rest)
train_groups = floor(SPLIT_RATIOS["train"] * full_group_count)
val_groups   = floor(SPLIT_RATIOS["val"]   * full_group_count)
test_groups  = full_group_count - train_groups - val_groups  # ensure all groups are used

# Index boundaries on groups (not images)
train_end = train_groups
val_end   = train_groups + val_groups
test_end  = full_group_count  # all remaining groups

assignments = (
    ("train", groups[:train_end]),
    ("val",   groups[train_end:val_end]),
    ("test",  groups[val_end:test_end]),
)

# Copy grouped images + labels
def copy_pair(img_path: Path, split: str):
    # image
    dst_img = Path(OUTPUT_DIR) / "images" / split / img_path.name
    shutil.copy2(img_path, dst_img)
    # label
    label_path = Path(LABELS_DIR) / f"{img_path.stem}.txt"
    if label_path.exists():
        dst_lbl = Path(OUTPUT_DIR) / "labels" / split / label_path.name
        shutil.copy2(label_path, dst_lbl)

for split, split_groups in assignments:
    for g in split_groups:
        for img_path in g:
            copy_pair(img_path, split)

# Put remainder (1–2 tail images) into TEST to avoid breaking any triplet
for img_path in remainder_imgs:
    copy_pair(img_path, "test")

# Summary
def count_objs(label_dir: Path) -> int:
    total = 0
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            total += sum(1 for _ in f)
    return total

print("\nSplit Summary:")
for split in ["train", "val", "test"]:
    img_dir = Path(OUTPUT_DIR) / "images" / split
    lbl_dir = Path(OUTPUT_DIR) / "labels" / split
    img_count = len(list(img_dir.glob("*.png")))
    obj_count = count_objs(lbl_dir)
    print(f"{split.capitalize()}: {img_count} images, {obj_count} objects")
    if split != "test":
        # sanity check: non-test splits should be multiples of GROUP_SIZE
        if img_count % GROUP_SIZE != 0:
            print(f"  [WARN] {split} image count ({img_count}) is not a multiple of {GROUP_SIZE}.")

# Extra info for debugging
print(f"\nTotal images: {n}")
print(f"Full groups: {full_group_count} (train={train_groups}, val={val_groups}, test={test_groups}), "
      f"remainder images sent to test: {len(remainder_imgs)}")
