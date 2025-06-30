import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

# Grade labels
GFONT = [
    '6A+','6B','6B+','6C','6C+',
    '7A','7A+','7B','7B+','7C',
    '7C+','8A','8A+','8B','8B+',
]
INT_TO_GRADE = {i: g for i, g in enumerate(GFONT)}

# Define dimensions for encoding
MAIN_DIMS = [f"{chr(ord('A')+c)}{r}" for c in range(11) for r in range(1,19)]
START_DIMS = [f"{chr(ord('A')+c)}{r}" for c in range(11) for r in range(1,7)]
END_DIMS = [f"{chr(ord('A')+c)}18" for c in range(13)]
POS2I_MAIN = {p: i for i, p in enumerate(MAIN_DIMS)}
POS2I_START = {p: i for i, p in enumerate(START_DIMS)}
POS2I_END = {p: i for i, p in enumerate(END_DIMS)}

# Load the trained model (ensure the model file path is correct)
MODEL_PATH = 'moonboard_model.keras'
model = load_model(MODEL_PATH)


def encode_holds(holds):
    """
    Encode a list of holds into model input vectors.
    Input holds are case insensitive, with optional 's' (start) or 't' (finish) suffix.
    """
    vec_main = np.zeros((1, len(MAIN_DIMS)), dtype=np.float32)
    vec_start = np.zeros((1, len(START_DIMS)), dtype=np.float32)
    vec_end = np.zeros((1, len(END_DIMS)), dtype=np.float32)

    for raw in holds:
        h = raw.strip().upper()
        is_start = False
        is_end = False
        if h.endswith('S'):
            is_start = True
            h = h[:-1]
        elif h.endswith('T'):
            is_end = True
            h = h[:-1]

        # Only consider valid positions
        if h in POS2I_MAIN:
            vec_main[0, POS2I_MAIN[h]] = 1
        if is_start and h in POS2I_START:
            vec_start[0, POS2I_START[h]] = 1
        if is_end and h in POS2I_END:
            vec_end[0, POS2I_END[h]] = 1

    # Concatenate features
    print('start',vec_start)
    print('end',vec_end)
    print('main',vec_main)
    return np.concatenate([vec_main, vec_start, vec_end], axis=1)


def predict_grade(holds):
    """
    Given a list of holds, returns a tuple (numeric_prediction, grade_label).
    Numeric prediction is a float; grade_label is the closest graded string.
    """
    x = encode_holds(holds)
    p = model.predict(x, verbose=0)[0, 0]
    # Round to nearest integer index, clip to valid range
    idx = int(np.clip(np.rint(p), 0, len(GFONT)-1))
    label = INT_TO_GRADE[idx]

# save and exit
    model_json = model.to_json()
    with open("moonboard_model.json", "w") as f:
        f.write(model_json)
    
    os.makedirs("weights", exist_ok=True)
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        for j, w in enumerate(weights):
            np.save(f"weights/layer_{i}_weight_{j}.npy", w)

    return p, label


def main():
    parser = argparse.ArgumentParser(
        description='Predict MoonBoard grade from holds sequence'
    )
    parser.add_argument(
        'holds', nargs='+', help="List of holds (e.g. B18 F6s D18t)"
    )
    args = parser.parse_args()
    pred_value, pred_label = predict_grade(args.holds)
    print('holds', args.holds)
    print(f"Predicted numeric grade: {pred_value:.2f}")
    print(f"Closest grade label : {pred_label}")


if __name__ == '__main__':
    main()
