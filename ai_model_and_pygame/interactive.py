
import sys
import pygame
import numpy as np
import scipy.ndimage
from PIL import Image
from skimage import measure
from collections import defaultdict
import torch
import json
from a import MoonModel

# ── Config ─────────────────────────────────────────────────────────────────────
MOON_IMG_PATH  = "moon.png"
STEP2_IMG_PATH = "step_2.png"
COLS, ROWS    = 11, 18
D_OUTLINE     = 10 
LABEL_PT      = 24
SELECT_PT     = 20
BOTTOM_PAD    = 0
RIGHT_PAD     = 200

# ── Load PyTorch model ─────────────────────────────────────────────────────────
# Assume you've saved a scripted or traced model at 'grade_model.pt'
model = new MoonModel()
model.eval()

def prepare_input(names, name_to_lab, ROWS, COLS):
    """
    Convert a list of hold names (e.g. ['A18', 'B5s', ...]) into a fixed-length tensor.
    Here we build a feature vector of length ROWS*COLS, setting values based on selection state.
    You may adapt this to match how your model was trained.
    """
    # Map suffixes to numeric codes
    code_map = {'on': 1.0, 'start': 2.0, 'top': 3.0}
    # Initialize feature vector
    x = torch.zeros(ROWS * COLS, dtype=torch.float32)
    for nm in names:
        # strip suffix to get base cell name and state suffix
        if nm.endswith('s'):
            base, state = nm[:-1], 'start'
        elif nm.endswith('t'):
            base, state = nm[:-1], 'top'
        else:
            base, state = nm, 'on'
        lab = name_to_lab.get(base)
        if lab is not None:
            idx = lab - 1  # zero-based index
            x[idx] = code_map[state]
    return x.unsqueeze(0)  # shape [1, ROWS*COLS]

# ── Load images & masks ─────────────────────────────────────────────────────────
step2 = np.array(Image.open(STEP2_IMG_PATH).convert("L")) > 0
H, W  = step2.shape
moon_np = np.array(Image.open(MOON_IMG_PATH).convert("RGBA"))
assert moon_np.shape[:2] == (H, W), "Size mismatch"

labels = measure.label(step2, connectivity=1)
ncomps = labels.max()

comp_coords = {lab: np.column_stack(np.where(labels==lab)) for lab in range(1, ncomps+1)}
comp_center = {lab: coords.mean(axis=0)[::-1] for lab, coords in comp_coords.items()}

# (centroid and naming logic unchanged)
# ... [snip] ...
# At end, generate ui_data and write JSON as before

# ── Inference loop ─────────────────────────────────────────────────────────────
while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            # ... handle selection toggling ...
            pass

    # ... rendering code unchanged ...

    # Build input tensor and run PyTorch inference
    names = sorted(name_with_suffix(lab_to_name(l), s) for l, s in selected.items())
    if names:
        # Prepare tensor
        x = prepare_input(names, name_to_lab, ROWS, COLS)
        with torch.no_grad():
            out = model(x)
        # Assuming model outputs a single float score and a class index
        g_float = out[0, 0].item()
        class_idx = int(out[0, 1].item())
        # Map class_idx to name, if needed
        g_name = inf.CLASS_NAMES[class_idx] if hasattr(inf, 'CLASS_NAMES') else str(class_idx)
        gradeTxt = f"Grade: {g_name} ({g_float:.2f})"
        gradeSurf = f_select.render(gradeTxt, True, (240,240,240))
        screen.blit(gradeSurf, (10, (m_top-5)*scale))

    pygame.display.flip()
    clock.tick(30)
