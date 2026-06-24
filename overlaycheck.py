import os
import random

import cv2

img_dir = "Kuartis_images"
label_dir = "Kuartis_labels"
output_dir = "debug_bbox"
os.makedirs(output_dir, exist_ok=True)

IMG_W, IMG_H = 2048, 2048

# Pick some random images to visualize
samples = random.sample([f for f in os.listdir(img_dir) if f.endswith(".png")], 5)

for img_name in samples:
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))

    # Load image
    img = cv2.imread(img_path)

    if not os.path.exists(label_path):
        print(f"No labels for {img_name}")
        continue

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())
        # Convert YOLO (normalized) back to pixels
        x1 = int((x - w / 2) * IMG_W)
        y1 = int((y - h / 2) * IMG_H)
        x2 = int((x + w / 2) * IMG_W)
        y2 = int((y + h / 2) * IMG_H)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save debug image
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, img)
    print(f"✅ Saved visualization: {out_path}")
