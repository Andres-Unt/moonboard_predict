import torch
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .pth to per-layer .npy.txt for TF.js")
    parser.add_argument("--model-path", required=True,
                        help="Path to moonboard_model.pth")
    parser.add_argument("--output-dir", default="weights",
                        help="Directory to write weights/*.npy.txt")
    args = parser.parse_args()

    # 1) Load
    state = torch.load(args.model_path, map_location="cpu")
    # If you saved the whole model, you might need state = state.state_dict()

    # 2) Grab only the Linear layers under `net.*.weight` / `net.*.bias`
    #    They occur at net[1], net[4], net[7], net[8] in your Sequential.
    #    Lexicographically, keys will be: net.1.weight, net.1.bias, net.4.weight, ...
    keys = [k for k in state.keys()
            if k.startswith("net.") and (k.endswith(".weight") or k.endswith(".bias"))]
    keys.sort(key=lambda k: (
        int(k.split('.')[1]),                # layer index in the nn.Sequential
        0 if k.endswith(".weight") else 1     # weight first, bias second
    ))

    os.makedirs(args.output_dir, exist_ok=True)

    # 3) Iterate and dump
    #    We'll assign a simple 0-based linear-layer counter
    layer_counter = -1
    prev_seq_idx = None

    for k in keys:
        seq_idx = int(k.split('.')[1])
        if seq_idx != prev_seq_idx:
            # new Linear layer
            layer_counter += 1
            prev_seq_idx = seq_idx

        param_type = "weight" if k.endswith(".weight") else "bias"
        param_idx = 0 if param_type == "weight" else 1

        tensor = state[k].cpu().numpy()

        # Build filename, e.g. "weights/layer_2_bias_1.npy.txt"
        fname = f"layer_{layer_counter}_{param_type}_{param_idx}.npy.txt"
        out_path = os.path.join(args.output_dir, fname)

        # Write as a true .npy container, but with a .txt extension
        # numpy.lib.format.write_array writes the header + data in .npy format
        with open(out_path, 'wb') as f:
            np.lib.format.write_array(f, tensor, allow_pickle=False)

        print(f"Wrote {out_path}  shape={tensor.shape}")

if __name__ == "__main__":
    main()

