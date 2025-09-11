import cv2
import numpy as np
import random

def radar_noise(image, prob=0.05, gauss_std=10, speckle_var=0.05):
    """
    Inject Gaussian + speckle noise to simulate radar clutter.
    - prob: probability to apply
    - gauss_std: standard deviation of Gaussian noise
    - speckle_var: variance of multiplicative speckle noise
    """
    import numpy as np

    if np.random.rand() < prob:
        # Gaussian noise
        gauss = np.random.normal(0, gauss_std, image.shape).astype(np.float32)

        # Speckle noise (multiplicative)
        speckle = image.astype(np.float32) * (1 + np.random.normal(0, speckle_var, image.shape))

        noisy = image.astype(np.float32) + gauss + speckle
        image = np.clip(noisy, 0, 255).astype(np.uint8)

    return image

def rcs_scaling(image, scale_range=(0.8, 1.2), prob=0.3):
    """
    Randomly scale pixel intensity to simulate Radar Cross Section (RCS) variation.
    Supports grayscale (H,W) and color (H,W,3).
    """
    if random.random() < prob:
        scale = random.uniform(*scale_range)
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return image


def azimuth_motion_blur(image, prob=0.2, kernel_size=5):
    """
    Apply horizontal motion blur to simulate azimuth smearing.
    Supports grayscale (H,W) and color (H,W,3).
    """
    if random.random() < prob:
        kernel = np.zeros((1, kernel_size))
        kernel[0] = np.ones(kernel_size) / kernel_size
        image = cv2.filter2D(image, -1, kernel)
    return image


def ghost_echo(image, prob=0.1, max_size=30, intensity=(0.2, 0.5)):
    """
    Add faint ghost patches to simulate multipath reflections or sea clutter.
    Supports grayscale (H,W) and color (H,W,3).
    """
    if random.random() < prob:
        h, w = image.shape[:2]

        # Choose random patch size and location
        patch_w, patch_h = random.randint(10, max_size), random.randint(10, max_size)
        x, y = random.randint(0, max(0, w - patch_w)), random.randint(0, max(0, h - patch_h))

        roi = image[y:y+patch_h, x:x+patch_w]

        # Create random ghost patch with same shape as ROI
        ghost_patch = np.random.randint(0, 100, roi.shape, dtype=np.uint8)

        # Blend ghost into ROI
        alpha = random.uniform(*intensity)
        blended = cv2.addWeighted(roi, 1.0, ghost_patch, alpha, 0)

        image[y:y+patch_h, x:x+patch_w] = blended

    return image


def range_distortion(image, prob=0.2, scale_range=(0.9, 1.1)):
    """
    Apply vertical scaling (range distortion) to simulate radar resolution effects.
    Supports grayscale (H,W) and color (H,W,3).
    """
    if random.random() < prob:
        h, w = image.shape[:2]
        scale = random.uniform(*scale_range)
        new_h = int(h * scale)

        # Resize vertically
        image_scaled = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad or crop to restore original size
        if new_h > h:
            image = image_scaled[:h, :]
        else:
            pad_h = h - new_h
            if image_scaled.ndim == 3:  # color (H,W,C)
                pad = np.zeros((pad_h, w, image_scaled.shape[2]), dtype=image.dtype)
            else:  # grayscale (H,W)
                pad = np.zeros((pad_h, w), dtype=image.dtype)
            image = np.vstack((image_scaled, pad))

    return image

