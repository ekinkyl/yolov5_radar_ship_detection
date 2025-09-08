import json
import os

json_dir = "jsons/"
total = 0
empty = 0
bad_files = []

for file in os.listdir(json_dir):
    if file.endswith(".json"):
        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)

        # Farklı json yapıları için kontrol:
        if isinstance(data, list):
            is_empty = len(data) == 0
        elif isinstance(data, dict) and "annotations" in data:
            is_empty = len(data["annotations"]) == 0
        else:
            is_empty = True  # tanınmayan yapı, boş kabul et
            bad_files.append(file)

        if is_empty:
            empty += 1
        total += 1

print(f"Toplam: {total}, Boş (objesiz): {empty}, Dolu: {total - empty}")
if bad_files:
    print(f"Uyarı: Şu dosyalar tanınmadı: {bad_files}")
