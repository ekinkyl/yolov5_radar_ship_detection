from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

# INPUT: sequential split
SRC_ROOT = Path("/raid/yolov5/moana_xband_gray_split_sequence_final")
# OUTPUT: new dataset with both streams
DST_ROOT = Path("/raid/yolov5/moana_xband_two_stream_bothS2")
SPLITS = ["train", "val", "test"]
S = 2  # temporal offset

def read_gray(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError(f"Cannot read {p}")
    return im

def ensure_dirs():
    for sp in SPLITS:
        # appearance stream dirs
        (DST_ROOT / "images_app" / sp).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / sp).mkdir(parents=True, exist_ok=True)
        # motion stream dirs
        (DST_ROOT / "images_mot" / sp).mkdir(parents=True, exist_ok=True)

def process_split(sp: str):
    src_img_dir = SRC_ROOT / "images" / sp
    src_lbl_dir = SRC_ROOT / "labels" / sp
    dst_app_dir = DST_ROOT / "images_app" / sp
    dst_mot_dir = DST_ROOT / "images_mot" / sp
    dst_lbl_dir = DST_ROOT / "labels" / sp

    imgs = sorted(src_img_dir.glob("*.png"))
    for i in tqdm(range(len(imgs)), desc=f"[{sp}] 3-frame two-stream"):
        cur  = imgs[i]
        prev = imgs[max(0, i - S)]
        nxt  = imgs[min(len(imgs) - 1, i + S)]

        im_prev = read_gray(prev).astype(np.float32)
        im_cur  = read_gray(cur).astype(np.float32)
        im_next = read_gray(nxt).astype(np.float32)

        # ---- Appearance stream (same as before) ----
        stacked_app = np.dstack([im_prev, im_cur, im_next]).astype(np.uint8)
        cv2.imwrite(str(dst_app_dir / cur.name), stacked_app)

        # ---- Motion stream (absolute differences) ----
        diff_prev = np.abs(im_cur - im_prev).astype(np.uint8)
        diff_next = np.abs(im_next - im_cur).astype(np.uint8)

        zeros = np.zeros_like(diff_prev, dtype=np.uint8)
        stacked_mot_rgb = np.dstack([diff_prev, diff_next, zeros])  # [H,W,3] uint8

        cv2.imwrite(str(dst_mot_dir / cur.name), stacked_mot_rgb)


        # ---- Labels (middle frame only) ----
        src_lbl = src_lbl_dir / (cur.stem + ".txt")
        dst_lbl = dst_lbl_dir / (cur.stem + ".txt")
        if src_lbl.exists():
            dst_lbl.write_text(src_lbl.read_text())

if __name__ == "__main__":
    ensure_dirs()
    for sp in SPLITS:
        process_split(sp)
    print("Done ->", DST_ROOT)
