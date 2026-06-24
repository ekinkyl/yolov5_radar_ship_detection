import os
import shutil

# Input datasets
moana_img_train = "/raid/yolov5/moana_xband_gray_split_sequence_final/images/train"
moana_lbl_train = "/raid/yolov5/moana_xband_gray_split_sequence_final/labels/train"

kuartis_img_train = "/raid/yolov5/Kuartis_split/images/train"
kuartis_lbl_train = "/raid/yolov5/Kuartis_split/labels/train"

kuartis_img_val = "/raid/yolov5/Kuartis_split/images/val"
kuartis_lbl_val = "/raid/yolov5/Kuartis_split/labels/val"

kuartis_img_test = "/raid/yolov5/Kuartis_split/images/test"
kuartis_lbl_test = "/raid/yolov5/Kuartis_split/labels/test"

# Output dataset
output_root = "merged_dataset"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)


def copy_all(src_imgs, src_lbls, dst_split):
    for img_file in sorted(os.listdir(src_imgs)):
        if not img_file.endswith(".png"):
            continue
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"

        # Copy image
        shutil.copy(os.path.join(src_imgs, img_file), os.path.join(output_root, "images", dst_split, img_file))

        # Copy label (if exists, else create empty file)
        src_lbl_path = os.path.join(src_lbls, label_file)
        dst_lbl_path = os.path.join(output_root, "labels", dst_split, label_file)

        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, dst_lbl_path)
        else:
            open(dst_lbl_path, "w").close()


# Build dataset
# Train = Moana train + Kuartis train
copy_all(moana_img_train, moana_lbl_train, "train")
copy_all(kuartis_img_train, kuartis_lbl_train, "train")

# Val = Kuartis val
copy_all(kuartis_img_val, kuartis_lbl_val, "val")

# Test = Kuartis test
copy_all(kuartis_img_test, kuartis_lbl_test, "test")

print("✅ Merged dataset created at:", output_root)
