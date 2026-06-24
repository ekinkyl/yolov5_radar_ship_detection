import os
import xml.etree.ElementTree as ET

xml_path = "annotations.xml"
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

# Default image size (if not explicitly in XML)
IMG_W, IMG_H = 2048, 2048

tree = ET.parse(xml_path)
root = tree.getroot()

for track in root.findall("track"):
    label = track.attrib["label"]  # e.g., "Dinamik Obje"
    cls_id = 0  # only one class

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

        # save label file per frame (e.g., 000851.txt)
        txt_name = f"{frame:06d}.txt"
        out_path = os.path.join(output_dir, txt_name)

        with open(out_path, "a") as f:
            f.write(line + "\n")

print("✅ Conversion finished! YOLO labels saved in:", output_dir)
