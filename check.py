import os
from pathlib import Path

# Root dataset folder
root = Path("/raid/yolov5/moana_xband_two_stream_both_s1")

splits = ["train", "val", "test"]

for split in splits:
    print(f"\n--- Checking {split} ---")

    # Paths
    img_dir = root / "images_app" / split
    lbl_dir = root / "labels" / split

    # Collect filenames without extensions
    img_stems = {f.stem for f in img_dir.glob("*.png")}
    lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

    # Compare
    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems

    print(f" Total images: {len(img_stems)}")
    print(f" Total labels: {len(lbl_stems)}")

    if missing_labels:
        print(f" Images without labels: {len(missing_labels)}")
        print(list(missing_labels)[:10])  # show first 10
    else:
        print(" ✅ All images have labels")

    if missing_images:
        print(f" Labels without images: {len(missing_images)}")
        print(list(missing_images)[:10])  # show first 10
    else:
        print(" ✅ All labels have images")

    # Quick check of empty labels
    empty = [f for f in lbl_dir.glob("*.txt") if f.stat().st_size == 0]
    if empty:
        print(f" ⚠️ Empty label files: {len(empty)}")
        print([e.name for e in empty[:10]])
    else:
        print(" ✅ No empty label files")
