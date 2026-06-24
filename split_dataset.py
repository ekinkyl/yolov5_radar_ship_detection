import os
import random
import shutil
from pathlib import Path

image_dir = Path("moana_xband/images")
label_dir = Path("moana_xband/labels")
output_base = Path("moana_xband")

image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
objeli = []
objesiz = []

for img_file in image_files:
    txt_file = img_file.replace(".png", ".txt")
    txt_path = label_dir / txt_file
    if not txt_path.exists():
        continue
    with open(txt_path) as f:
        lines = f.readlines()
    if len(lines) == 0:
        objesiz.append((img_file, 0))
    else:
        objeli.append((img_file, len(lines)))


random.shuffle(objeli)
random.shuffle(objesiz)


def split(lst):
    n = len(lst)
    return lst[: int(0.8 * n)], lst[int(0.8 * n) : int(0.9 * n)], lst[int(0.9 * n) :]


train_o, val_o, test_o = split(objeli)
train_e, val_e, test_e = split(objesiz)

train = train_o + train_e
val = val_o + val_e
test = test_o + test_e


def copy_split(split_data, split_name):
    img_out = output_base / "images" / split_name
    lbl_out = output_base / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_obj = 0
    for img_file, obj_count in split_data:
        src_img = image_dir / img_file
        src_lbl = label_dir / img_file.replace(".png", ".txt")

        dst_img = img_out / img_file
        dst_lbl = lbl_out / src_lbl.name

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        total_obj += obj_count

    print(f"{split_name.upper()} -> {len(split_data)} images, {total_obj} objects")


copy_split(train, "train")
copy_split(val, "val")
copy_split(test, "test")
