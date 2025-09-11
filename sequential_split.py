import os
import shutil
from pathlib import Path

# ===== CONFIG =====
GRAY_IMAGES_DIR = Path("gray_images/images")   # tüm grayscale imgeler
LABELS_ROOT = Path("moana_xband_gray_split_sequence_final/labels")       # split edilmiş label klasörleri
OUTPUT_ROOT = Path("moana_xband_gray_split_sequence_final")
# ==================

def copy_images_for_split(split: str):
    lbl_dir = LABELS_ROOT / split
    img_dir = OUTPUT_ROOT / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    count, missing = 0, 0
    for lbl_file in lbl_dir.glob("*.txt"):
        stem = lbl_file.stem  # örn: X_1724240642
        for ext in [".png", ".jpg", ".jpeg"]:
            src_img = GRAY_IMAGES_DIR / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, img_dir / src_img.name)
                count += 1
                break
        else:
            print(f"[WARN] No grayscale image found for {stem}")
            missing += 1

    print(f"{split}: {count} images copied, {missing} missing")

for split in ["train", "val", "test"]:
    copy_images_for_split(split)
