import os

# Folders to check
folders = [
    "moana_xband_gray_split_sequence_final/images",
    "moana_xband_gray_split_sequence_final/labels",
    "Kuartis_Split/images",
    "Kuartis_Split/labels",
    "newDataset/images",
    "newDataset/labels",
]

# Loop through each folder
for folder in folders:
    print(f"\n📂 {folder}")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(folder, split)
        if os.path.exists(split_path):
            count = len([f for f in os.listdir(split_path) if os.path.isfile(os.path.join(split_path, f))])
            print(f"  {split}: {count} files")
        else:
            print(f"  {split}: ❌ not found")
