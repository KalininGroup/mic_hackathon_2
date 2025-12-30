import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import match_template
from skimage.transform import rescale

# ==============================
# USER INPUTS
# # ==============================
# IMG2_PATH = r'C:\Users\Aadarsh\Desktop\Random_works\Hackathon\Images\Stiching\Polymer\1.png'
# IMG1_PATH = r'C:\Users\Aadarsh\Desktop\Random_works\Hackathon\Images\Stiching\Polymer\7.png'


IMG2_PATH = r'C:\Users\Aadarsh\Desktop\Random_works\Hackathon\Images\Stiching\Stars\1.png'
IMG1_PATH = r'C:\Users\Aadarsh\Desktop\Random_works\Hackathon\Images\Stiching\Stars\11.png'


SCAN1_UM = 10.0
SCAN2_UM = 10.0

GRID_N = 4
SCALE_RANGE = np.linspace(0.7, 1.5, 15)
TOP_N_GRIDS = 3  # number of grids to use for anchor consensus

# ==============================
# LOAD + PREPROCESS
# ==============================
img1 = imread(IMG1_PATH)
img2 = imread(IMG2_PATH)

if img1.ndim == 3:
    img1 = img1[:, :, 0]
if img2.ndim == 3:
    img2 = img2[:, :, 0]

img1 = img1.astype(float)
img2 = img2.astype(float)

def normalize(img):
    img = img - np.median(img)
    return img / np.std(img)

img1 = normalize(img1)
img2 = normalize(img2)

# ==============================
# PIXEL → NM SCALING
# ==============================
nm_px_1 = (SCAN1_UM * 500) / img1.shape[1]
nm_px_2 = (SCAN2_UM * 500) / img2.shape[1]
physical_scale = nm_px_1 / nm_px_2

# ==============================
# DIVIDE IMAGE 1 INTO GRID
# ==============================
h, w = img1.shape
gh, gw = h // GRID_N, w // GRID_N

grid_matches = []

for i in range(GRID_N):
    for j in range(GRID_N):
        y0, y1 = i * gh, (i + 1) * gh
        x0, x1 = j * gw, (j + 1) * gw
        template = img1[y0:y1, x0:x1]

        best_score = -np.inf
        best_px, best_py, best_scale = None, None, None

        for s in SCALE_RANGE * physical_scale:
            tmpl = rescale(template, s, preserve_range=True, anti_aliasing=True)
            if tmpl.shape[0] >= img2.shape[0] or tmpl.shape[1] >= img2.shape[1]:
                continue
            result = match_template(img2, tmpl)
            score = result.max()
            if score > best_score:
                py, px = np.unravel_index(np.argmax(result), result.shape)
                best_score = score
                best_px, best_py, best_scale = px, py, s

        grid_matches.append({
            "grid_idx": (i, j),
            "origin": (x0, y0),
            "template": template,
            "score": best_score,
            "match_xy": (best_px, best_py),
            "scale": best_scale
        })

# ==============================
# SELECT TOP N GRIDS & COMPUTE ANCHOR
# ==============================
grid_matches.sort(key=lambda x: x["score"], reverse=True)
top_grids = grid_matches[:TOP_N_GRIDS]

dxs, dys = [], []
for g in top_grids:
    x0, y0 = g["origin"]
    px2, py2 = g["match_xy"]
    dxs.append(px2 - x0)
    dys.append(py2 - y0)

dx_avg = int(np.round(np.mean(dxs)))
dy_avg = int(np.round(np.mean(dys)))

print(f"Average anchor shift (Image1 → Image2): dx = {dx_avg}, dy = {dy_avg}")
print(f"Shift in nm: dx = {dx_avg*nm_px_1:.1f}, dy = {dy_avg*nm_px_1:.1f}")

# ==============================
# COMPUTE MERGE CANVAS
# ==============================
# Treat Image2 as fixed at (0,0)
# Image1 offset to align grids
x1_offset = dx_avg
y1_offset = dy_avg

# Handle negative offsets
x_shift = max(-x1_offset, 0)
y_shift = max(-y1_offset, 0)

# Final canvas size
canvas_w = max(img2.shape[1] + x_shift, img1.shape[1] + x1_offset + x_shift)
canvas_h = max(img2.shape[0] + y_shift, img1.shape[0] + y1_offset + y_shift)

canvas = np.zeros((canvas_h, canvas_w))
weight = np.zeros_like(canvas)

# Place Image2 at (x_shift, y_shift)
img2_x = x_shift
img2_y = y_shift
canvas[img2_y:img2_y + img2.shape[0], img2_x:img2_x + img2.shape[1]] += img2
weight[img2_y:img2_y + img2.shape[0], img2_x:img2_x + img2.shape[1]] += 1

# Place Image1 at (x1_offset + x_shift, y1_offset + y_shift)
img1_x = x1_offset + x_shift
img1_y = y1_offset + y_shift
canvas[img1_y:img1_y + img1.shape[0], img1_x:img1_x + img1.shape[1]] += img1
weight[img1_y:img1_y + img1.shape[0], img1_x:img1_x + img1.shape[1]] += 1

merged = canvas / np.maximum(weight,1)

# ==============================
# LOCATE GRID POSITIONS IN IMAGE2
# ==============================
grid_in_img2 = []
for g in top_grids:
    px2, py2 = g["match_xy"]
    grid_in_img2.append({
        "grid_idx": g["grid_idx"],
        "position": (px2, py2),
        "size": g["template"].shape
    })

# ==============================
# VISUALIZATION
# ==============================
# ==============================
# VISUALIZATION
# ==============================
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Image1 with grid
axs[0].imshow(img1, cmap='gray')
axs[0].set_title("Image 1 with grids")
axs[0].axis('off')
for g in top_grids:
    x0, y0 = g["origin"]
    h, w = g["template"].shape
    rect = plt.Rectangle((x0, y0), w, h, edgecolor='r', facecolor='none', linewidth=1.5)
    axs[0].add_patch(rect)

# Image2 with located grids
axs[1].imshow(img2, cmap='gray')
axs[1].set_title("Image 2 with located grids similarity")
axs[1].axis('off')
for g in grid_in_img2:
    px, py = g["position"]
    h, w = g["size"]
    rect = plt.Rectangle((px, py), w, h, edgecolor='g', facecolor='none', linewidth=1.5)
    axs[1].add_patch(rect)

# Top template
axs[2].imshow(top_grids[0]["template"], cmap='gray')
axs[2].set_title(f"Grid with Highest similarity coefficient template {top_grids[0]['grid_idx']}")
axs[2].axis('off')

# Merged image
axs[3].imshow(merged, cmap='gray')
axs[3].set_title(f"Merged image (anchored around grid {top_grids[0]['grid_idx']})")
axs[3].axis('off')

# Add drift/shifts at bottom
dx_nm = dx_avg * nm_px_1
dy_nm = dy_avg * nm_px_1
plt.figtext(0.5, 0.01, f"Drift/Shifts: dx = {dx_avg} px ({dx_nm:.1f} nm), dy = {dy_avg} px ({dy_nm:.1f} nm)",
            ha="center", fontsize=12, color="blue")

plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave space at bottom for note
plt.show()

