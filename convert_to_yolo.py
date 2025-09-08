import json
import os

json_dir = "jsons"
output_dir = "moana_xband/labels"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)

        image_w = data["image"]["width"]
        image_h = data["image"]["height"]
        annotations = data.get("annotations", [])

        yolo_lines = []
        for ann in annotations:
            xmin = ann["xmin"]
            ymin = ann["ymin"]
            w = ann["width"]
            h = ann["height"]

            x_center = (xmin + w / 2) / image_w
            y_center = (ymin + h / 2) / image_h
            norm_w = w / image_w
            norm_h = h / image_h

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        txt_name = file.replace(".json", ".txt")
        with open(os.path.join(output_dir, txt_name), "w") as out:
            out.write("\n".join(yolo_lines))
