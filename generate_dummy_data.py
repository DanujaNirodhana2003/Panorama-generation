"""
generate_dummy_data.py
======================
Generate a synthetic 4-frame panoramic sequence for testing the multi-image
pipeline and bundle adjustment without real data.

Scene: wide canvas with a red square, green circle, blue triangle, and text.
Images: 4 overlapping crops at ~37% overlap each (left→right sweep).

Output: data/raw/frame_00.jpg … frame_03.jpg  (sorted filenames → correct order)
"""

import cv2
import numpy as np
import os

# Canvas (400 × 1200 px — wide enough for 4 crops with overlap)
W, H = 1200, 400
canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas[:] = (200, 200, 200)

# Rich, distinctive scene content so SIFT finds plenty of features
cv2.rectangle(canvas, (50,  80), (250, 320), (0,   0, 255), -1)   # red square
cv2.rectangle(canvas, (50,  80), (250, 320), (0,   0, 128),  4)   # dark border
cv2.circle   (canvas, (400, 200), 120, (0, 200,  0),  -1)          # green circle
cv2.circle   (canvas, (400, 200),  80, (0, 100,  0),  -1)          # darker inner
cv2.fillPoly (canvas, [np.array([[700, 60],[580, 340],[820, 340]])], (255, 0, 0))  # blue triangle
cv2.rectangle(canvas, (950, 80), (1150, 320), (0, 200, 200), -1)   # cyan square
cv2.putText  (canvas, "Panorama G16", (300, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.putText  (canvas, "M4 Test Seq",  (750, 380),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)

# 4 overlapping crops: stride = 250 px, width = 500 px → 50% overlap
CROP_W  = 500
STRIDE  = 250
N_FRAMES = 4

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data", "raw")
os.makedirs(data_dir, exist_ok=True)

for i in range(N_FRAMES):
    x0 = i * STRIDE
    x1 = x0 + CROP_W
    frame = canvas[:, x0:x1].copy()
    fname = os.path.join(data_dir, f"frame_{i:02d}.jpg")
    cv2.imwrite(fname, frame)
    print(f"Saved: {fname}  (x: {x0}–{x1})")

print(f"\nGenerated {N_FRAMES} overlapping frames in: {data_dir}")
print("Run: python src/pipeline.py --multi")
print("Run: python src/pipeline.py --multi --ba  (with bundle adjustment)")
