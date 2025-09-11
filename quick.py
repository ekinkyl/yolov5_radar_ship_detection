import os

def count_files(img_dir, lbl_dir):
    imgs = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    return len(imgs), len(lbls)

# Paths
moana_train_img = "/raid/yolov5/moana_xband_gray_split_sequence_final/images/train"
moana_train_lbl = "/raid/yolov5/moana_xband_gray_split_sequence_final/labels/train"

kuartis_train_img = "/raid/yolov5/Kuartis_split/images/train"
kuartis_train_lbl = "/raid/yolov5/Kuartis_split/labels/train"

kuartis_val_img = "/raid/yolov5/Kuartis_split/images/val"
kuartis_val_lbl = "/raid/yolov5/Kuartis_split/labels/val"

kuartis_test_img = "/raid/yolov5/Kuartis_split/images/test"
kuartis_test_lbl = "/raid/yolov5/Kuartis_split/labels/test"

merged_train_img = "merged_dataset/images/train"
merged_train_lbl = "merged_dataset/labels/train"

merged_val_img = "merged_dataset/images/val"
merged_val_lbl = "merged_dataset/labels/val"

merged_test_img = "merged_dataset/images/test"
merged_test_lbl = "merged_dataset/labels/test"

# Count and print
print("📊 File Counts")
print("-" * 40)

print("Moana Train:", count_files(moana_train_img, moana_train_lbl))
print("Kuartis Train:", count_files(kuartis_train_img, kuartis_train_lbl))
print("Merged Train:", count_files(merged_train_img, merged_train_lbl))

print("Kuartis Val:", count_files(kuartis_val_img, kuartis_val_lbl))
print("Merged Val:", count_files(merged_val_img, merged_val_lbl))

print("Kuartis Test:", count_files(kuartis_test_img, kuartis_test_lbl))
print("Merged Test:", count_files(merged_test_img, merged_test_lbl))
