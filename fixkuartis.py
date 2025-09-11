import os

labels_root = "/raid/yolov5/newDataset/labels"
output_root = "/raid/yolov5/newDataset/labels_fixed"
os.makedirs(output_root, exist_ok=True)

splits = ["train", "val", "test"]

# Old canvas size (CVAT export assumption)
W_old = 4096  # guessed from comparison
H_old = 2048  # height seems fine
W_new = 2048
H_new = 2048

for split in splits:
    in_dir = os.path.join(labels_root, split)
    out_dir = os.path.join(output_root, split)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.endswith(".txt"):
            continue
        
        fixed_lines = []
        with open(os.path.join(in_dir, fname)) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())

                # Step 1: back to pixels using CVAT's assumed old size
                x_pix = x * W_old
                y_pix = y * H_old
                w_pix = w * W_old
                h_pix = h * H_old

                # Step 2: re-normalize to true image size
                x_new = x_pix / W_new
                y_new = y_pix / H_new
                w_new = w_pix / W_new
                h_new = h_pix / H_new

                fixed_lines.append(
                    f"{int(cls)} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}\n"
                )

        with open(os.path.join(out_dir, fname), "w") as f:
            f.writelines(fixed_lines)

print("✅ Fixed labels saved in:", output_root)
