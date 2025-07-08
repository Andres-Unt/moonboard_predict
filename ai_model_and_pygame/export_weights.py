import torch
import numpy as np
import os

# Load PyTorch model state dict
state = torch.load('moonboard_conv.pth', map_location='cpu')

# Create output dir
out_dir = 'weights'
os.makedirs(out_dir, exist_ok=True)

# Iterate and save each tensor as .npy
for name, tensor in state.items():
    arr = tensor.cpu().numpy()
    # sanitize name for filename
    fname = name.replace('.', '_') + '.npy'
    path = os.path.join(out_dir, fname)
    np.save(path, arr)
    os.rename(path, path.replace('.npy', '.npy.txt'))
    print(f"Saved {name} -> {path} (shape={arr.shape})")
