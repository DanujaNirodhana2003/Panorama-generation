"""
src/evaluation/metrics.py
=========================
Evaluation metrics: RMSE, inlier ratio, seam intensity metric.

Extracted from src/evaluate_M3.py (now superseded by this module).

M3 Baseline reference metrics (image1.jpeg ↔ image2.jpeg):
  - Total SIFT matches  : varies (typically 200–400 after ratio test)
  - Inlier ratio        : 76.56 %
  - RMSE (inliers only) : 0.54 px
  - Wall time           : ~0.65 s/pair
"""

import cv2
import numpy as np
import time


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_rmse(src_pts: np.ndarray, dst_pts: np.ndarray,
                 H: np.ndarray) -> float:
    """
    Compute reprojection RMSE of src_pts → dst_pts through homography H.

    Parameters
    ----------
    src_pts : ndarray (N, 1, 2) or (N, 2)  – source keypoints
    dst_pts : ndarray (N, 1, 2) or (N, 2)  – destination keypoints
    H       : ndarray (3, 3)                – homography matrix

    Returns
    -------
    rmse : float  – root-mean-square reprojection error in pixels
    """
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(warped - dst, axis=2)
    return float(np.sqrt(np.mean(errors ** 2)))


def inlier_stats(mask: np.ndarray, total_matches: int) -> dict:
    """
    Compute inlier count and ratio from a RANSAC mask.

    Parameters
    ----------
    mask          : ndarray  – output mask from findHomography (1=inlier, 0=outlier)
    total_matches : int      – total good matches before RANSAC

    Returns
    -------
    dict with keys: 'inliers', 'total', 'ratio' (0.0–1.0), 'ratio_pct' (%)
    """
    n_inliers = int(mask.ravel().sum()) if mask is not None else 0
    ratio = n_inliers / max(total_matches, 1)
    return {
        "inliers":   n_inliers,
        "total":     total_matches,
        "ratio":     ratio,
        "ratio_pct": ratio * 100.0,
    }


def seam_intensity_metric(panorama: np.ndarray,
                          seam_x: int | None = None,
                          window: int = 10) -> float:
    """
    Qualitative seam visibility metric: measures the mean absolute gradient
    magnitude in a vertical strip around the estimated seam line.

    A lower value means a less visible seam.

    Parameters
    ----------
    panorama : ndarray (H, W, C)  – stitched image
    seam_x   : int | None         – x-coordinate of the seam (estimated from
                                    image centre if None)
    window   : int                – half-width of the strip around seam_x

    Returns
    -------
    seam_score : float  – mean gradient magnitude near seam (lower = better)
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) if panorama.ndim == 3 else panorama
    if seam_x is None:
        seam_x = gray.shape[1] // 2
    x0 = max(0, seam_x - window)
    x1 = min(gray.shape[1], seam_x + window)
    strip = gray[:, x0:x1].astype(np.float32)
    grad_x = np.abs(np.diff(strip, axis=1))
    return float(grad_x.mean())


# ---------------------------------------------------------------------------
# Full pair evaluation (run feature extraction + RANSAC + report)
# ---------------------------------------------------------------------------

def evaluate_pair(img1_path: str, img2_path: str,
                  ratio_thresh: float = 0.70) -> dict:
    """
    Run the full SIFT+FLANN+RANSAC pipeline on an image pair and return
    all M3/M4 metrics.

    Returns
    -------
    dict with keys: total_matches, inliers, inlier_ratio_pct, rmse_px, time_s
    """
    t0 = time.perf_counter()

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not load images: {img1_path}, {img2_path}")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
    )
    all_matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in all_matches if m.distance < ratio_thresh * n.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    t1 = time.perf_counter()

    stats = inlier_stats(mask, len(good))
    rmse  = float("nan")
    if H is not None and mask is not None:
        inlier_src = src_pts[mask.ravel() == 1]
        inlier_dst = dst_pts[mask.ravel() == 1]
        rmse = compute_rmse(inlier_src, inlier_dst, H)

    return {
        "total_matches":    len(good),
        "inliers":          stats["inliers"],
        "inlier_ratio_pct": stats["ratio_pct"],
        "rmse_px":          rmse,
        "time_s":           t1 - t0,
    }
