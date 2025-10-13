import os

import cv2

# Root dirs
root_labels = "/raid/yolov5/newDataset/labels"
root_images = "/raid/yolov5/newDataset/images"
output_root = "/raid/yolov5/newDataset/labels_fixed"

splits = ["train", "val", "test"]

for split in splits:
    labels_dir = os.path.join(root_labels, split)
    images_dir = os.path.join(root_images, split)
    output_dir = os.path.join(output_root, split)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔄 Processing {split} split...")

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, fname)
        image_path = os.path.join(images_dir, fname.replace(".txt", ".png"))  # adjust if .jpg
        if not os.path.exists(image_path):
            print(f"⚠️ Image missing for {label_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Cannot read {image_path}")
            continue
        H, W = img.shape[:2]

        fixed_lines = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)

                # Convert normalized -> pixels
                x_pix = x * W
                y_pix = y * H
                w_pix = w * W
                h_pix = h * H

                # Re-normalize correctly
                x_new = x_pix / W
                y_new = y_pix / H
                w_new = w_pix / W
                h_new = h_pix / H

                fixed_lines.append(f"{int(cls)} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}\n")

        # Save corrected file
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w") as f:
            f.writelines(fixed_lines)

    print(f"✅ Finished {split}, saved to {output_dir}")

print("🎯 All splits processed. Fixed labels in:", output_root)
