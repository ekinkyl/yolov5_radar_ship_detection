#!/usr/bin/env python3
"""
Overlay TP (green), FP (red), and FN (yellow) on test images.

Assumptions:
- Test images:   moana_xband_gray_split/images/test/*.png
- GT labels:     moana_xband_gray_split/labels/test/*.txt   (YOLO format)
- Predictions:   runs/val/xband_test_results/labels/*.txt   (YOLO format from val.py --save-txt)

Run:  python visualize_fp_fn.py
"""

import glob
from pathlib import Path

import cv2

# ---- paths (edit if needed) ----
ROOT = Path("/raid/yolov5")
IMG_DIR = ROOT / "moana_xband_gray_split_sequence_final/images/test"
GT_DIR = ROOT / "moana_xband_gray_split_sequence_final/labels/test"
PRED_DIR = ROOT / "runs/val/xband_s_ddp_b32_g3g417_test/labels"
OUT_DIR = ROOT / "runs/val/xband_s_ddp_b32_g3g417_test/vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- colors ----
COL_TP = (0, 255, 0)  # green
COL_FP = (0, 0, 255)  # red
COL_FN = (0, 255, 255)  # yellow

IOU_THR = 0.7  # match threshold


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt_boxes(txt_path, W, H):
    """YOLO GT: class xc yc w h (normalized)."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # ignore class id for drawing
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append(yolo_to_xyxy(xc, yc, w, h, W, H))
    return boxes


def load_pred_boxes(txt_path, W, H):
    """YOLO preds (from val.py --save-txt): class xc yc w h [conf]
    Be tolerant to presence/absence/order of conf.
    """
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # common format: cls xc yc w h conf
            cls, xc, yc, w, h = map(float, parts[:5])
            # if there's a 6th value, it's confidence
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            boxes.append(((x1, y1, x2, y2), conf))
    return boxes


# iterate images
pngs = sorted(glob.glob(str(IMG_DIR / "*.png")))
for ip, img_path in enumerate(pngs, 1):
    img_path = Path(img_path)
    name = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    gt_path = GT_DIR / f"{name}.txt"
    pred_path = PRED_DIR / f"{name}.txt"

    gt_boxes = load_gt_boxes(gt_path, W, H)
    pred_boxes = load_pred_boxes(pred_path, W, H)  # list of ((x1,y1,x2,y2), conf)

    matched_gt = set()
    matched_pred = set()

    # greedy matching: for each pred, find best GT over IOU_THR
    for pi, (pb, conf) in enumerate(pred_boxes):
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            i = iou(pb, gb)
            if i > best_iou:
                best_iou, best_gi = i, gi
        if best_iou >= IOU_THR:
            matched_pred.add(pi)
            matched_gt.add(best_gi)
            # TP (green)
            x1, y1, x2, y2 = pb
            cv2.rectangle(img, (x1, y1), (x2, y2), COL_TP, 2)
            cv2.putText(
                img, f"TP {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TP, 1, cv2.LINE_AA
            )

    # FP (red): predictions not matched
    for pi, (pb, conf) in enumerate(pred_boxes):
        if pi in matched_pred:
            continue
        x1, y1, x2, y2 = pb
        cv2.rectangle(img, (x1, y1), (x2, y2), COL_FP, 2)
        cv2.putText(img, f"FP {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_FP, 1, cv2.LINE_AA)

    # FN (yellow): GT not matched
    for gi, gb in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        x1, y1, x2, y2 = gb
        cv2.rectangle(img, (x1, y1), (x2, y2), COL_FN, 2)
        cv2.putText(img, "FN", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_FN, 1, cv2.LINE_AA)

    out_path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(out_path), img)

print(f"✅ Done. Visuals saved to: {OUT_DIR}")
