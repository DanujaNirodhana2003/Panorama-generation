"""
src/geometry/homography.py
==========================
RANSAC homography estimation with M4 skew/stretch fallback.

M3 Baseline reference
---------------------
cv2.findHomography(..., cv2.RANSAC, 5.0) achieved:
  - Inlier ratio : 76.56 %
  - RMSE         : 0.54 px

M4 Addition
-----------
After RANSAC, the resulting warp's bounding-box is measured. If either the
output width or height exceeds STRETCH_LIMIT × the original dimension, we fall
back to a constrained affine transform (estimateAffinePartial2D) which only
allows rotation + uniform scale + translation — far less likely to catastrophically
distort small-overlap image pairs. Every fallback event is logged to a CSV.
"""

import csv
import os
import time
import cv2
import numpy as np

# Default stretch factor beyond which we fall back to affine
STRETCH_LIMIT = 1.3


def _bounding_box_stretch(img_shape, H):
    """
    Warp the four corners of an image through H and measure the resulting
    bounding-box dimensions relative to the original image dimensions.

    Returns
    -------
    w_ratio : float   – output_width  / original_width
    h_ratio : float   – output_height / original_height
    """
    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped  = cv2.perspectiveTransform(corners, H)
    x_min, y_min = warped[:, 0, :].min(axis=0)
    x_max, y_max = warped[:, 0, :].max(axis=0)
    return (x_max - x_min) / w, (y_max - y_min) / h


def estimate_homography(src_pts, dst_pts, img2_shape,
                        ransac_reproj_thresh: float = 5.0,
                        stretch_limit: float = STRETCH_LIMIT,
                        log_path: str | None = None,
                        pair_label: str = "unknown"):
    """
    Estimate a homography (or affine fallback) from point correspondences.

    Parameters
    ----------
    src_pts              : ndarray (N, 1, 2)  – source points (img1 keypoints)
    dst_pts              : ndarray (N, 1, 2)  – destination points (img2 keypoints)
    img2_shape           : tuple              – shape of the image being warped (h, w[, c])
    ransac_reproj_thresh : float              – RANSAC reprojection threshold (px)
    stretch_limit        : float              – ratio above which affine fallback triggers
    log_path             : str | None         – path to fallback log CSV (None = no log)
    pair_label           : str               – human-readable name for this pair (for logging)

    Returns
    -------
    H        : ndarray (3, 3) or None  – homography (or lifted affine, or None on failure)
    mask     : ndarray (N,)            – inlier mask (1 = inlier)
    fallback : bool                    – True if affine fallback was used
    """
    fallback = False

    # Guard against too few points (< 4 cannot define a homography)
    if src_pts is None or dst_pts is None or len(src_pts) < 4 or len(dst_pts) < 4:
        return None, None, False

    # --- Step 1: RANSAC homography -------------------------------------------
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)

    if H is None:
        return None, None, False

    # --- Step 2: Check for catastrophic stretch ------------------------------
    w_ratio, h_ratio = _bounding_box_stretch(img2_shape, H)
    max_stretch = max(w_ratio, h_ratio)

    if max_stretch > stretch_limit:
        fallback = True
        axis = "width" if w_ratio > h_ratio else "height"
        reason = (f"stretch {max_stretch:.2f}× > limit {stretch_limit}× "
                  f"(axis: {axis})")
        print(f"  [FALLBACK] {pair_label}: {reason}. "
              f"Using affine instead of full homography.")

        # --- Step 3: Affine fallback (rotation + scale + translation only) ---
        A, mask_a = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_thresh
        )
        if A is None:
            print(f"  [FALLBACK] Affine also failed for {pair_label}. Returning None.")
            return None, None, True

        # Lift 2×3 affine to 3×3 homography
        H = np.vstack([A, [0, 0, 1]])
        mask = mask_a.ravel() if mask_a is not None else mask

        # --- Step 4: Log fallback event --------------------------------------
        if log_path is not None:
            _log_fallback(log_path, pair_label, max_stretch, stretch_limit,
                          w_ratio, h_ratio, reason)

    return H, mask, fallback


def _log_fallback(log_path: str, pair_label: str, max_stretch: float,
                  stretch_limit: float, w_ratio: float, h_ratio: float,
                  reason: str):
    """Append one fallback event row to a CSV log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "pair", "w_ratio", "h_ratio",
                             "max_stretch", "stretch_limit", "reason"])
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            pair_label,
            f"{w_ratio:.3f}",
            f"{h_ratio:.3f}",
            f"{max_stretch:.3f}",
            f"{stretch_limit:.2f}",
            reason,
        ])
