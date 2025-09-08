#!/usr/bin/env python3
"""
Bulk-convert X-band dataset images to grayscale.

- Walks the entire --input directory tree.
- Converts .png/.jpg/.jpeg images to grayscale.
- Preserves relative folder structure in --output.
- Optional: copies labels/ files as-is.
- Handles images with alpha channels.
"""

import argparse
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def to_grayscale(img_bgr, mode="3ch"):
    """
    Convert BGR (or BGRA) image to grayscale.

    mode: '1ch' -> single channel, '3ch' -> 3-channel grayscale (recommended).
    """
    if img_bgr is None:
        return None

    # If BGRA, drop alpha first
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "1ch":
        return gray
    # default: 3-channel grayscale
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def should_copy(path: Path):
    # copy non-image files (e.g., labels) when requested
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input root folder (e.g., data/moana_xband)")
    ap.add_argument("--output", required=True, help="Output root folder (e.g., data/moana_xband_gray)")
    ap.add_argument("--mode", choices=["1ch", "3ch"], default="3ch", help="Grayscale save mode")
    ap.add_argument(
        "--copy-labels",
        action="store_true",
        help="If set, non-image files under any 'labels' directory are copied as-is.",
    )
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    count_img, count_skip, count_copy = 0, 0, 0

    for src_path in in_root.rglob("*"):
        rel = src_path.relative_to(in_root)
        dst_path = out_root / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        ext = src_path.suffix.lower()

        # Convert images
        if ext in IMG_EXTS:
            # Read unchanged to handle alpha if present
            img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Could not read image: {src_path}")
                count_skip += 1
                continue

            gray_img = to_grayscale(img, mode=args.mode)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(dst_path), gray_img)
            if not ok:
                print(f"[WARN] Failed to write: {dst_path}")
                count_skip += 1
            else:
                count_img += 1

        # Optionally mirror labels/ or other non-image files
        elif args.copy - labels and "labels" in [p.name for p in src_path.parents]:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            count_copy += 1

    print("\nDone.")
    print(f"Converted images : {count_img}")
    print(f"Copied label files: {count_copy}")
    print(f"Skipped/failed    : {count_skip}")
    print(f"Output root       : {out_root}")


if __name__ == "__main__":
    main()
