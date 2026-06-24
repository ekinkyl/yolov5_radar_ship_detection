import os
from pathlib import Path

import cv2
import numpy as np

# === CONFIG ===
baseline_dir = Path("runs/val/baseline_merged_test/vis")
compare_dir = Path("runs/val/final_train_merged_test/vis")
output_dir = Path("runs/val/compare_merged_baseline_vs_final_train_test")
output_dir.mkdir(parents=True, exist_ok=True)

label_left = "Baseline"
label_right = "ClusterNMS"

# === PROCESS ===
for file in os.listdir(baseline_dir):
    baseline_img_path = baseline_dir / file
    compare_img_path = compare_dir / file

    if compare_img_path.exists():
        img_left = cv2.imread(str(baseline_img_path))
        img_right = cv2.imread(str(compare_img_path))

        if img_left is None or img_right is None:
            print(f"⚠️ Skipping {file}, could not read image")
            continue

        # make heights equal
        h = min(img_left.shape[0], img_right.shape[0])
        img_left = cv2.resize(img_left, (int(img_left.shape[1] * h / img_left.shape[0]), h))
        img_right = cv2.resize(img_right, (int(img_right.shape[1] * h / img_right.shape[0]), h))

        # add a white separator (20px wide)
        separator = 255 * np.ones((h, 20, 3), dtype=np.uint8)

        # put labels on images (black text)
        cv2.putText(img_left, label_left, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_right, label_right, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

        # concatenate [left | separator | right]
        combined = np.hstack([img_left, separator, img_right])

        # save
        cv2.imwrite(str(output_dir / file), combined)
        print(f"✅ Saved comparison: {file}")

print("All done! Comparisons are in:", output_dir)
