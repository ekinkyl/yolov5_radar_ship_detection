from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

# INPUT: your sequential split
SRC_ROOT = Path("/raid/yolov5/moana_xband_gray_split_sequence_final")
# OUTPUT: new dataset with 3‑channel temporal images
DST_ROOT = Path("/raid/yolov5/moana_xband_gray_temporal_2")
SPLITS = ["train", "val", "test"]
S = 2  # temporal offset (adjacent frames)

def read_gray(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Cannot read {p}")
    return im

def ensure_dirs():
    for sp in SPLITS:
        (DST_ROOT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / sp).mkdir(parents=True, exist_ok=True)

def process_split(sp: str):
    src_img_dir = SRC_ROOT / "images" / sp
    src_lbl_dir = SRC_ROOT / "labels" / sp
    dst_img_dir = DST_ROOT / "images" / sp
    dst_lbl_dir = DST_ROOT / "labels" / sp

    imgs = sorted(src_img_dir.glob("*.png"))
    for i in tqdm(range(len(imgs)), desc=f"[{sp}] 3‑frame"):
        cur  = imgs[i]
        prev = imgs[max(0, i - S)]
        nxt  = imgs[min(len(imgs) - 1, i + S)]

        im_prev = read_gray(prev)
        im_cur  = read_gray(cur)
        im_next = read_gray(nxt)

        stacked = np.dstack([im_prev, im_cur, im_next])  # [prev, cur, next]
        cv2.imwrite(str(dst_img_dir / cur.name), stacked)

        # copy label of the MIDDLE frame (cur)
        src_lbl = src_lbl_dir / (cur.stem + ".txt")
        dst_lbl = dst_lbl_dir / (cur.stem + ".txt")
        if src_lbl.exists():
            dst_lbl.write_text(src_lbl.read_text())

if __name__ == "__main__":
    ensure_dirs()
    for sp in SPLITS:
        process_split(sp)
    print("Done ->", DST_ROOT)
