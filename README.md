# Marine Radar Ship Detection Based on YOLOv5 with Integrated YOSMR Modules

This repository contains an advanced computer vision framework for robust ship detection in marine radar images. It integrates custom YOSMR (Customized Lightweight Convolutional Networks) modules and architectural adaptations into the YOLOv5 framework to address classic radar perception challenges, such as wave clutter, land/shore echoes, and weak reflections from small vessels.

## Features & Key Modules

The project adapts standard object detection for specialized X-band marine radar imagery through the integration of the following modules:

* **MobileNetV3 (Large) Backbone:** Replaces the standard YOLOv5 backbone with a lightweight network utilizing depthwise separable convolutions and Squeeze-and-Excitation (SE) attention modules to maximize efficiency for edge-compute hardware.
* **LightPANet Neck:** A customized feature fusion architecture where standard C3 blocks are optimized into lightweight C3Light structures and convolution channels are systematically halved.
* **YOSMR Spatial Pyramid Pooling (SPP):** Unlike YOLOv5's sequential SPPF, this utilizes 4 independent, parallel branches ($1\times1$, $5\times5$, $9\times9$, $13\times13$) to capture a richer spectrum of multi-scale details.
* **Cluster-NMS:** Replaces traditional Non-Maximum Suppression (NMS). Instead of discarding overlapping candidate boxes, it merges them intelligently based on inference confidence to improve recall.
* **$\alpha$-DIoU Loss:** An optimized bounding box regression loss function adding adaptive power scaling ($\alpha$) to center distance metrics, improving localization for small or blurry targets.
* **Temporal Context (Single Stream):** Supports stacking 3 consecutive radar frames (sampled at frames $t-2, t, t+2$) into a multi-channel tensor to capture short motion history.

---

## Dataset & Preprocessing

The model was trained and evaluated on the open-source **MOANA X-Band Radar dataset** alongside proprietary radar datasets provided by **Kuartis**.

### Preprocessing Pipeline:
1. **Grayscale Conversion:** Standard RGB inputs are transformed into uniform 3-channel grayscale arrays.
2. **Label Translation:** Custom parser utility mapping standard dataset JSON files containing absolute bounding boxes `(xmin, ymin, width, height)` to standard normalized YOLO `.txt` files `(class, x_center, y_center, width, height)`.
3. **Dataset Split:** Split into **70% Training**, **10% Validation**, and **20% Testing**.

### Specialized Radar Augmentations:
To mitigate domain-specific environmental noise, a custom augmentation module injects the following transformations:
* Noise injection (Gaussian & Speckle)
* RCS (Radar Cross Section) scaling
* Azimuth and range distortion mapping
* Ghost echo simulation

---

## Training Configuration

Models are trained using PyTorch under the following operational hyperparameters:

```yaml
Optimizer: SGD (lr0=0.005, momentum=0.937, weight_decay=0.0005)
Epochs: 150 (Early stopping patience set to 20)
Batch Size: 32
Loss Balancing Coeffs: box=0.05, obj=1.2, cls=0.5
