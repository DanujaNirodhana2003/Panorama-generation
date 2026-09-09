"""
src/blending/multiband.py
=========================
Laplacian pyramid multi-band blending (M4 — higher-quality blend mode).

M3 Baseline issue
-----------------
Feathering (src/blending/feathering.py) smooths exposure seams but can soften
high-frequency details (edges, structural lines) in the overlap region because
the same spatially-varying weight is applied to ALL frequency bands.

M4 Fix
------
Multi-band blending (Burt & Adelson, 1983) builds a Gaussian pyramid of the
blend mask and a Laplacian pyramid of each image, blends each level separately,
and then reconstructs. Low-frequency bands are blended over a wide transition
zone (removes exposure seams); high-frequency bands use a sharp boundary
(preserves edge detail). Result: seamless AND sharp panoramas.

Selectable via configs/default.yaml: blend_mode: multiband
"""

import cv2
import numpy as np


def _build_gaussian_pyramid(img: np.ndarray, levels: int):
    """Build a Gaussian pyramid (list of downsampled images)."""
    pyramid = [img.copy()]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def _build_laplacian_pyramid(img: np.ndarray, levels: int):
    """Build a Laplacian pyramid from a Gaussian pyramid."""
    gp = _build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        expanded = cv2.pyrUp(gp[i + 1], dstsize=size)
        lap = cv2.subtract(gp[i].astype(np.float32),
                           expanded.astype(np.float32))
        lp.append(lap)
    lp.append(gp[levels].astype(np.float32))  # coarsest level (base)
    return lp


def _reconstruct_from_laplacian(lp):
    """Collapse a Laplacian pyramid back to an image."""
    img = lp[-1]
    for level in reversed(lp[:-1]):
        size = (level.shape[1], level.shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = img + level
    return img


def multiband_blend(canvas1: np.ndarray, canvas2: np.ndarray,
                    mask1: np.ndarray, mask2: np.ndarray,
                    levels: int = 5) -> np.ndarray:
    """
    Multi-band Laplacian pyramid blending.

    Parameters
    ----------
    canvas1, canvas2 : ndarray (H, W, C), float32  – images on the common canvas
    mask1,   mask2   : ndarray (H, W),    uint8    – binary masks (255=filled)
    levels           : int                          – pyramid depth (default 5)

    Returns
    -------
    blended : ndarray (H, W, C), uint8
    """
    # Build smooth blend mask from distance-transform weights
    d1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3).astype(np.float32)
    d2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3).astype(np.float32)
    total = d1 + d2 + 1e-6
    w1_map = (d1 / total)  # shape (H, W)

    # Expand mask to 3 channels for per-channel pyramid ops
    w1_3c = np.stack([w1_map] * canvas1.shape[2], axis=-1)

    # Build Laplacian pyramids for both images and the blend mask
    lp1 = _build_laplacian_pyramid(canvas1, levels)
    lp2 = _build_laplacian_pyramid(canvas2, levels)
    gp_mask = _build_gaussian_pyramid(w1_3c, levels)

    # Blend each pyramid level
    blended_lp = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        # Resize mask level to match image level size (pyrDown can differ by ±1)
        gm_resized = cv2.resize(gm, (l1.shape[1], l1.shape[0]))
        blended_level = gm_resized * l1 + (1.0 - gm_resized) * l2
        blended_lp.append(blended_level)

    # Reconstruct
    result = _reconstruct_from_laplacian(blended_lp)

    # Fill regions covered by only one image (no blending needed there)
    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask1 == 0) & (mask2 > 0)
    result[only1] = canvas1[only1]
    result[only2] = canvas2[only2]

    return np.clip(result, 0, 255).astype(np.uint8)
