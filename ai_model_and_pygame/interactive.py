import sys
import pygame
import numpy as np
import scipy.ndimage
from PIL import Image
from skimage import measure
from collections import defaultdict
import grade_inference as inf
import json

# ── Config ─────────────────────────────────────────────────────────────────────
MOON_IMG_PATH  = "moon.png"
STEP2_IMG_PATH = "step_2.png"
COLS, ROWS    = 11, 18
D_OUTLINE     = 10 
LABEL_PT      = 24
SELECT_PT     = 20
BOTTOM_PAD    = 0
RIGHT_PAD     = 200

# ── Load images & masks ─────────────────────────────────────────────────────────
step2 = np.array(Image.open(STEP2_IMG_PATH).convert("L")) > 0
H, W  = step2.shape
moon_np = np.array(Image.open(MOON_IMG_PATH).convert("RGBA"))
assert moon_np.shape[:2] == (H, W), "Size mismatch"

labels = measure.label(step2, connectivity=1)
ncomps = labels.max()

comp_coords = {lab: np.column_stack(np.where(labels==lab)) for lab in range(1, ncomps+1)}
comp_center = {lab: coords.mean(axis=0)[::-1] for lab, coords in comp_coords.items()}

centroids = list(comp_center.items())
centroids.sort(key=lambda t: (t[1][1], t[1][0]))

name_to_lab = {}
for row_idx in range(ROWS):
    slice_ = centroids[row_idx*COLS:(row_idx+1)*COLS]
    slice_.sort(key=lambda t: t[1][0])
    row_label = ROWS - row_idx
    for col_idx, (lab, _) in enumerate(slice_):
        name = f"{chr(ord('A')+col_idx)}{row_label}"
        name_to_lab[name] = lab

col_x = defaultdict(list)
row_y = defaultdict(list)
cell_w = W/COLS
cell_h = H/ROWS

for name, lab in name_to_lab.items():
    col = ord(name[0]) - ord('A')
    row = ROWS - int(name[1:])
    x, y = comp_center[lab]
    col_x[col].append(x)
    row_y[row].append(y)

X_col = {c: np.mean(xs) for c, xs in col_x.items()}
Y_row = {r: np.mean(ys) for r, ys in row_y.items()}

m_left, m_top = cell_w, cell_h
canvas_w = m_left + W + RIGHT_PAD
canvas_h = m_top + H + BOTTOM_PAD

pygame.init()
info = pygame.display.Info()
max_w, max_h = info.current_w * 0.9, info.current_h * 0.9
scale = min(1.0, max_w/canvas_w, max_h/canvas_h)
disp = (int(canvas_w*scale), int(canvas_h*scale))
screen = pygame.display.set_mode(disp)
pygame.display.set_caption("Hold Selector")

moon_surf = pygame.image.frombuffer(moon_np.tobytes(), (W, H), "RGBA")
moon_surf = pygame.transform.smoothscale(moon_surf, (int(W*scale), int(H*scale)))
f_label  = pygame.font.SysFont(None, int(LABEL_PT*2*scale))
f_select = pygame.font.SysFont(None, int(SELECT_PT*3*scale))

def image_xy(mx, my):
    x = mx/scale - m_left
    y = my/scale - m_top
    xi, yi = int(x), int(y)
    if 0 <= xi < W and 0 <= yi < H:
        return xi, yi
    return None

def lab_to_name(lab):
    for n, L in name_to_lab.items():
        if L == lab:
            return n
    return str(lab)


def name_with_suffix(name, state):
    if state == "on": return name
    if state == "start": return name + "s"
    if state == "top": return name + "t"
    return None

selected = {}
clock = pygame.time.Clock()



# … your existing imports and mask/centroid computation …

# At the end of your script, instead of dumping the raw arrays:
ui_data = {
    "W": int(W),
    "H": int(H),
    "COLS": COLS,
    "ROWS": ROWS,
    "m_left": float(cell_w),
    "m_top": float(cell_h),
    "RIGHT_PAD": float(RIGHT_PAD),
    "BOTTOM_PAD": float(BOTTOM_PAD),
    # name_to_lab: e.g. {"A18": 123, …}
    "name_to_lab": name_to_lab,
    # comp_center: convert each numpy array to [x, y]
    "comp_center": {lab: [float(x), float(y)]
                    for lab, (x, y) in comp_center.items()},
    # X_col, Y_row: already floats
    "X_col": {str(k): float(v) for k, v in X_col.items()},
    "Y_row": {str(k): float(v) for k, v in Y_row.items()},
    "lab_to_name": {str(lab): name for name, lab in name_to_lab.items()},
}

with open("public/ui_map.json", "w") as f:
    json.dump(ui_data, f, indent=2)


while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            pt = image_xy(*ev.pos)
            if pt:
                lab = labels[pt[1], pt[0]]
                if lab > 0:
                    name = lab_to_name(lab)
                    row = int(name[1:])
                    cur = selected.get(lab, None)

                    if row == 18:
                        states = ["top", "on", None]
                    elif 1 <= row <= 6:
                        states = ["on", "start", None]
                    else:
                        states = ["on", None]

                    idx = (states.index(cur) + 1) % len(states) if cur in states else 0
                    nxt = states[idx]

                    if nxt is None:
                        selected.pop(lab, None)
                    else:
                        selected[lab] = nxt

    screen.fill((30,30,30))

    for lab, state in selected.items():
        comp = (labels==lab)
        dil  = scipy.ndimage.binary_dilation(comp, iterations=D_OUTLINE)
        border = dil & ~comp & ~step2
        ys, xs = np.where(border)

        if state == "start":
            color = (0, 255, 0)  # Green
        elif state == "top":
            color = (255, 0, 0)  # Red
        else:
            color = (0, 120, 255)  # Blue

        for y, x in zip(ys, xs):
            sx = int((m_left + x)*scale)
            sy = int((m_top  + y)*scale)
            screen.set_at((sx, sy), color)

    screen.blit(moon_surf, (m_left*scale, m_top*scale))

    for c in range(COLS):
        letter = chr(ord('A')+c)
        x0, y0 = X_col[c], Y_row[0]
        vx, vy = x0 - X_col[c], y0 - Y_row[1]
        lx, ly = x0 + vx, y0 + vy
        surf = f_label.render(letter, True, (200,200,200))
        screen.blit(surf, ((m_left+lx)*scale - surf.get_width()/2, (m_top +ly)*scale - surf.get_height()/2))

    for r in range(ROWS):
        num = str(ROWS-r)
        x0, y0 = X_col[0], Y_row[r]
        vx, vy = X_col[1] - X_col[0], Y_row[r] - Y_row[r]
        lx, ly = x0 - vx, y0
        surf = f_label.render(num, True, (200,200,200))
        screen.blit(surf, ((m_left+lx)*scale - surf.get_width()/2, (m_top +ly)*scale - surf.get_height()/2))

    names = sorted(name_with_suffix(lab_to_name(l), s) for l, s in selected.items())
    txt = "Selected: " + ", ".join(names)
    surf = f_select.render(txt, True, (240,240,240))
    screen.blit(surf, (10, (m_top-55)*scale))

    grade = inf.predict_grade(names) if names else ""
    if grade:
        g_float, g_name = grade
        gradeTxt = f"Grade: {g_name} ({g_float:.2f})"
        gradeSurf = f_select.render(gradeTxt, True, (240,240,240))
        screen.blit(gradeSurf, (10, (m_top-5)*scale))

    pygame.display.flip()
    clock.tick(30)
