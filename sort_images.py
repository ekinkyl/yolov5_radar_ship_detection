import os
import xml.etree.ElementTree as ET

xml_path = "annotations.xml"
img_dir = "Kuartis_Dataset"
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

IMG_W, IMG_H = 2048, 2048

# Sorted list of actual image filenames
images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

tree = ET.parse(xml_path)
root = tree.getroot()

for track in root.findall("track"):
    cls_id = 0
    for box in track.findall("box"):
        frame = int(box.attrib["frame"])
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])

        w = xbr - xtl
        h = ybr - ytl
        x_center = (xtl + w / 2) / IMG_W
        y_center = (ytl + h / 2) / IMG_H
        norm_w = w / IMG_W
        norm_h = h / IMG_H

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

        # Map frame index to actual image filename
        if frame < len(images):
            txt_name = os.path.splitext(images[frame])[0] + ".txt"
            out_path = os.path.join(output_dir, txt_name)

            with open(out_path, "a") as f:
                f.write(line + "\n")

            # Debug print for confirmation
            print(f"Frame {frame} → {images[frame]} → {txt_name}")

print("✅ Conversion done. YOLO labels saved in:", output_dir)
