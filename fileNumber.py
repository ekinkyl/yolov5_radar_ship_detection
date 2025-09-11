import os
from pathlib import Path

# Paths
DATASET_ROOT = Path("/raid/yolov5/moana_xband_gray_split_sequence_final/images")

for split in ["train", "val", "test"]:
    split_dir = DATASET_ROOT / split
    # Count files with image extensions
    count = sum(1 for f in split_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"])
    print(f"{split}: {count} images")
