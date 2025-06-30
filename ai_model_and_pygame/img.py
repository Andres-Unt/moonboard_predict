from PIL import Image
import numpy as np
from skimage import measure
from skimage.draw import polygon
import cv2
import os

# === Step 0: Convert moon17.jpg to moon17.png ===
img = Image.open("moon17.jpg")
img.save("moon17.png")

# === Step 1: Create step_1.png from mask.png ===
mask = Image.open("mask.png").convert("RGBA")
mask_np = np.array(mask)

# A pixel is "non-color" if fully transparent or all RGB channels are 0
non_empty = np.any(mask_np[:, :, :3] != 0, axis=2) & (mask_np[:, :, 3] > 0)
step1 = (non_empty * 255).astype(np.uint8)
Image.fromarray(step1).save("step_1.png")

import scipy.ndimage

DILATION_RADIUS = 7

# Dilate entire step1 mask to merge close components
dilated_mask = scipy.ndimage.binary_dilation(step1 > 0, iterations=DILATION_RADIUS)

# Label connected components on the dilated mask
labels = measure.label(dilated_mask, connectivity=1)
step2 = np.zeros_like(step1, dtype=np.uint8)

for region_label in range(1, labels.max() + 1):
    region_mask = (labels == region_label)

    coords = np.column_stack(np.where(region_mask))
    if coords.shape[0] < 3:
        continue

    # Convex hull of this merged region
    hull = cv2.convexHull(coords.astype(np.int32))

    # Create empty mask and fill hull
    hull_mask = np.zeros_like(step1, dtype=np.uint8)
    rr, cc = polygon(hull[:, 0, 0], hull[:, 0, 1], hull_mask.shape)
    hull_mask[rr, cc] = 1

    # Erode hull mask to shrink it back by dilation radius
    eroded = scipy.ndimage.binary_erosion(hull_mask, iterations=DILATION_RADIUS)

    # Add eroded hull to final mask
    step2[eroded] = 255

Image.fromarray(step2).save("step_2.png")

# === Step 3: Mask moon17.png with step_2.png to get moon.png ===
moon_color = Image.open("moon17.png").convert("RGBA")
moon_np = np.array(moon_color)

# Resize step2 if needed to match moon17.png
if moon_np.shape[:2] != step2.shape:
    step2 = cv2.resize(step2, (moon_np.shape[1], moon_np.shape[0]), interpolation=cv2.INTER_NEAREST)

# Apply mask: if step2 is 255 (white), keep moon17.png color; otherwise black
mask_3ch = step2[:, :, None] // 255  # shape: (H, W, 1)
moon_np_masked = moon_np * mask_3ch

Image.fromarray(moon_np_masked).save("moon.png")

print("All steps completed. Files saved:")
print(" - moon17.png")
print(" - step_1.png")
print(" - step_2.png")
print(" - moon.png")

