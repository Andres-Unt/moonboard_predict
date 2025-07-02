
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
state_path = "moonboard_model.pth"
try:
    # Attempt to load a full saved model or state dict
    state = torch.load(state_path, map_location='cpu')
except Exception as e:
    print(f"Failed to load model file: {e}")
    sys.exit(1)

if isinstance(state, torch.nn.Module):
    # User saved entire model
    model = state
elif isinstance(state, dict):
    # User saved state_dict or a dict containing state_dict
    # If it contains a 'model_state_dict' key, extract it
    sd = state.get('model_state_dict', state)
    # Dynamically infer architecture? Attempt load with strict=False
    # User must define GradeNet matching this state_dict elsewhere if necessary
    model = MoonModel()
    model.load_state_dict(sd)
else:
    print("Unrecognized model format in .pth file.")
    sys.exit(1)

model.eval()

# ── Inference helper ────────────────────────────────────────────────────────────
def prepare_input(names, name_to_lab, ROWS, COLS):
    code_map = {'on': 1.0, 'start': 2.0, 'top': 3.0}
    x = torch.zeros(ROWS * COLS, dtype=torch.float32)
    for nm in names:
        if nm.endswith('s'):
            base, state = nm[:-1], 'start'
        elif nm.endswith('t'):
            base, state = nm[:-1], 'top'
        else:
            base, state = nm, 'on'
        lab = name_to_lab.get(base)
        if lab is not None:
            x[lab - 1] = code_map[state]
    return x.unsqueeze(0)

# ── Load UI mapping ────────────────────────────────────────────────────────────
with open("public/ui_map.json") as f:
    ui = json.load(f)
name_to_lab = ui['name_to_lab']

# CLI entrypoint
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python grade_inference.py <hold1> [hold2] ...")
        sys.exit(1)
    names = [nm.upper() for nm in sys.argv[1:]]
    x = prepare_input(names, name_to_lab, ROWS, COLS)
    with torch.no_grad():
        out = model(x) if hasattr(model, 'forward') else model(x)
    # Handle outputs flexibly
    if out.ndim == 2 and out.size(1) >= 1:
        score = out[0, 0].item()
        class_idx = int(out[0, 1].item()) if out.size(1) > 1 else None
    elif out.ndim == 1:
        score = out[0].item()
        class_idx = None
    else:
        print("Unexpected model output shape:", out.shape)
        sys.exit(1)

    # Map class_idx to name if available
    try:
        from grade_inference import CLASS_NAMES
        g_name = CLASS_NAMES[class_idx] if class_idx is not None else ''
    except ImportError:
        g_name = str(class_idx) if class_idx is not None else ''

    if class_idx is not None:
        print(f"Grade: {g_name} ({score:.2f})")
    else:
        print(f"Score: {score:.2f}")
