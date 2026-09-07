"""
src/blending/feathering.py
==========================
Distance-based alpha feathering blending (M4 — fast default blend mode).

M3 Baseline issue
-----------------
The original hard-overwrite compositing creates a sharp boundary between img1
and warped img2, producing visible exposure seams (especially when handheld
shots have slightly different exposure/white-balance settings).

M4 Fix
------
Each pixel in the overlap region is blended as a weighted average:

    out = (w1 * canvas1 + w2 * warped2) / (w1 + w2)

where w1, w2 are distance-transform weights (distance from nearest image
boundary), so the blend weight transitions smoothly across the seam zone.
This eliminates the hard exposure jump without softening high-frequency edges.
"""

import cv2
import numpy as np


def feather_blend(canvas1: np.ndarray, canvas2: np.ndarray,
                  mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Blend two canvases using distance-transform feathering.

    Parameters
    ----------
    canvas1, canvas2 : ndarray (H, W, C), float32  – images on the common canvas
    mask1,   mask2   : ndarray (H, W),    uint8    – binary masks (255=filled)

    Returns
    -------
    blended : ndarray (H, W, C), uint8
    """
    # Distance weights
    d1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3).astype(np.float32)
    d2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3).astype(np.float32)
    total = d1 + d2 + 1e-6  # avoid divide-by-zero

    # Expand to (H, W, 1) for broadcasting across colour channels
    w1 = (d1 / total)[..., np.newaxis]
    w2 = (d2 / total)[..., np.newaxis]

    # Only-img1 region, only-img2 region, and overlap
    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask1 == 0) & (mask2 > 0)
    both  = (mask1 > 0) & (mask2 > 0)

    out = np.zeros_like(canvas1, dtype=np.float32)
    out[only1] = canvas1[only1]
    out[only2] = canvas2[only2]
    out[both]  = (w1 * canvas1 + w2 * canvas2)[both]

    return np.clip(out, 0, 255).astype(np.uint8)
