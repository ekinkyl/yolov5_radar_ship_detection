# check_weights.py
import argparse

import torch
from ultralytics.utils.patches import torch_load

from models.yolo import Model
from utils.downloads import attempt_download
from utils.general import LOGGER


def check_weights(cfg, weights, device="cpu", nc=1):
    """
    Load a YOLOv5 model from cfg and try to apply weights with strict=False.

    Prints which layers matched, which are missing, and which are unexpected.
    """
    device = torch.device(device)

    # 1. Create fresh model from your cfg
    model = Model(cfg, ch=3, nc=nc).to(device)

    # 2. Load checkpoint
    ckpt_path = attempt_download(weights)
    ckpt = torch_load(ckpt_path, map_location="cpu")
    ckpt_model = (ckpt.get("ema") or ckpt["model"]).float()

    # 3. Apply weights with strict=False
    missing, unexpected = model.load_state_dict(ckpt_model.state_dict(), strict=False)

    LOGGER.info(f"\n⚡ Weight loading report for {cfg} with {weights}")
    LOGGER.info(f"   Missing keys: {len(missing)}")
    if missing:
        for k in missing[:30]:
            LOGGER.info(f"      MISSING: {k}")

    LOGGER.info(f"   Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:30]:
            LOGGER.info(f"      UNEXPECTED: {k}")

    LOGGER.info("\n✅ Done. Matching layers got pretrained weights.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to model yaml (e.g. models/yosmr.yaml)")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights file (e.g. yolov5l.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    opt = parser.parse_args()

    check_weights(opt.cfg, opt.weights, device=opt.device, nc=opt.nc)
