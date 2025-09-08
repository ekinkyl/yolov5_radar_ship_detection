import os
import shutil
from math import floor
from pathlib import Path

# ===== CONFIG =====
IMAGES_DIR = "moana_xband_gray/images"  # grayscale images folder
LABELS_DIR = "moana_xband/labels"  # label folder (YOLO format)
OUTPUT_DIR = "moana_xband_gray_split_sequence_2"  # output base folder

SPLIT_RATIOS = {"train": 0.8, "test": 0.1, "val": 0.1}  # train -> test -> val
GROUP_SIZE = 3  # keep 3 consecutive frames together
REMAINDER_TARGET_SPLIT = "val"  # send tail images here to avoid breaking triplets in train/test
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
remainder_imgs = images[full_group_count * GROUP_SIZE :]  # 0, 1, or 2 images
groups = [images[i * GROUP_SIZE : (i + 1) * GROUP_SIZE] for i in range(full_group_count)]

# Compute number of groups per split (train/test by floor, val gets the rest)
train_groups = floor(SPLIT_RATIOS["train"] * full_group_count)
test_groups = floor(SPLIT_RATIOS["test"] * full_group_count)
val_groups = full_group_count - train_groups - test_groups  # ensure all groups are used

# Index boundaries on groups (not images) — order: TRAIN -> TEST -> VAL
train_end = train_groups
test_end = train_groups + test_groups
val_end = full_group_count

assignments = (
    ("train", groups[:train_end]),
    ("test", groups[train_end:test_end]),
    ("val", groups[test_end:val_end]),
)


def copy_pair(img_path: Path, split: str):
    # image
    dst_img = Path(OUTPUT_DIR) / "images" / split / img_path.name
    shutil.copy2(img_path, dst_img)
    # label
    label_path = Path(LABELS_DIR) / f"{img_path.stem}.txt"
    if label_path.exists():
        dst_lbl = Path(OUTPUT_DIR) / "labels" / split / label_path.name
        shutil.copy2(label_path, dst_lbl)


# Copy grouped images + labels per split
for split, split_groups in assignments:
    for g in split_groups:
        for img_path in g:
            copy_pair(img_path, split)

# Put remainder (1–2 tail images) into the last split (VAL by default)
for img_path in remainder_imgs:
    copy_pair(img_path, REMAINDER_TARGET_SPLIT)


# Summary
def count_objs(label_dir: Path) -> int:
    total = 0
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            total += sum(1 for _ in f)
    return total


print("\nSplit Summary:")
for split in ["train", "test", "val"]:
    img_dir = Path(OUTPUT_DIR) / "images" / split
    lbl_dir = Path(OUTPUT_DIR) / "labels" / split
    img_count = len(list(img_dir.glob("*.png")))
    obj_count = count_objs(lbl_dir)
    print(f"{split.capitalize()}: {img_count} images, {obj_count} objects")

    # sanity checks: train & test should be multiples of GROUP_SIZE (remainder goes to val)
    if split in ["train", "test"] and img_count % GROUP_SIZE != 0:
        print(f"  [WARN] {split} image count ({img_count}) is not a multiple of {GROUP_SIZE}.")

print(f"\nTotal images: {n}")
print(
    f"Full groups: {full_group_count} (train={train_groups}, test={test_groups}, val={val_groups}), "
    f"remainder images sent to {REMAINDER_TARGET_SPLIT}: {len(remainder_imgs)}"
)
