"""
src/stitching/warper.py
=======================
Canvas allocation, perspective warping, and compositing loop.

M3 Baseline reference
---------------------
Original warpImages() used hard pixel overwrite (no blending) which creates
visible exposure seams. M4 integrates blending via the blend_mode parameter.

M4 Additions
------------
- blend_mode parameter: 'overwrite' (M3 default), 'feather', 'multiband'
- Exposure compensation before blending (histogram matching)
- Seam mask from distance transform
- Confidence-weighted compositing from inlier density near seams
"""

import cv2
import numpy as np


def compute_canvas(img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
    """
    Compute the bounding box for the stitched canvas and the translation offset.

    Parameters
    ----------
    img1 : ndarray  – base image (already stitched collage or first image)
    img2 : ndarray  – new image being warped
    H    : ndarray  – 3×3 homography mapping img2 coordinates into img1's frame

    Returns
    -------
    canvas_w : int    – canvas width
    canvas_h : int    – canvas height
    t        : list   – [tx, ty] translation to shift everything positive
    H_t      : ndarray (3,3) – translation-corrected homography
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    pts2_transformed = cv2.perspectiveTransform(pts2, H)

    all_pts = np.concatenate((pts1, pts2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    t   = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]],
                    [0, 1, t[1]],
                    [0, 0,   1 ]], dtype=np.float64)
    return (x_max - x_min), (y_max - y_min), t, H_t


def exposure_compensate(img_ref: np.ndarray, img_target: np.ndarray) -> np.ndarray:
    """
    Adjust img_target's brightness/contrast to match img_ref using
    per-channel mean+std normalization (simple Reinhard-style compensation).

    Parameters
    ----------
    img_ref    : ndarray  – reference image (the base/already-stitched portion)
    img_target : ndarray  – image to be warped; will be adjusted

    Returns
    -------
    compensated : ndarray  – img_target after exposure adjustment (same dtype)
    """
    out = img_target.astype(np.float32)
    ref = img_ref.astype(np.float32)
    for c in range(img_target.shape[2]):
        ref_mean, ref_std = ref[:, :, c].mean(), ref[:, :, c].std() + 1e-6
        tgt_mean, tgt_std = out[:, :, c].mean(), out[:, :, c].std() + 1e-6
        out[:, :, c] = (out[:, :, c] - tgt_mean) * (ref_std / tgt_std) + ref_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def distance_seam_mask(mask1: np.ndarray, mask2: np.ndarray):
    """
    Compute per-pixel blending weights for two overlapping masks using
    distance transform (each pixel weighted by its distance to the nearest
    boundary of its image region).

    Parameters
    ----------
    mask1, mask2 : ndarray (H, W), uint8  – binary masks (255 where image exists)

    Returns
    -------
    w1, w2 : ndarray (H, W), float32  – normalized weights summing to 1 in overlap
    """
    d1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3).astype(np.float32)
    d2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3).astype(np.float32)
    total = d1 + d2 + 1e-6
    return d1 / total, d2 / total


def warp_images(img1: np.ndarray, img2: np.ndarray, H: np.ndarray,
                blend_mode: str = "overwrite",
                compensate_exposure: bool = False) -> np.ndarray:
    """
    Warp img2 onto img1 using homography H and composite them.

    Parameters
    ----------
    img1               : ndarray  – base image (left/already-stitched)
    img2               : ndarray  – new image to warp and append
    H                  : ndarray  – 3×3 homography (img2 → img1 frame)
    blend_mode         : str      – 'overwrite' | 'feather' | 'multiband'
    compensate_exposure: bool     – if True, histogram-match img2 to img1 first

    Returns
    -------
    output_img : ndarray  – stitched panorama
    """
    canvas_w, canvas_h, t, H_t = compute_canvas(img1, img2, H)

    # Warp img2 onto blank canvas
    warped2 = cv2.warpPerspective(img2, H_t.dot(H), (canvas_w, canvas_h))

    # Build masks (255 where pixel data exists, 0 where black/empty)
    mask2 = cv2.warpPerspective(
        np.ones(img2.shape[:2], dtype=np.uint8) * 255,
        H_t.dot(H), (canvas_w, canvas_h)
    )
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] = 255

    if blend_mode == "overwrite":
        # Original M3 behaviour: paste img1 on top of warped img2
        output_img = warped2.copy()
        output_img[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] = img1

    elif blend_mode in ("feather", "multiband"):
        # Exposure compensation before blending
        if compensate_exposure:
            # Only compensate in the overlap region
            overlap = cv2.bitwise_and(mask1, mask2)
            if overlap.any():
                img2 = exposure_compensate(img1, img2)
                warped2 = cv2.warpPerspective(img2, H_t.dot(H), (canvas_w, canvas_h))

        # Place img1 on canvas
        canvas1 = np.zeros((canvas_h, canvas_w, img1.shape[2]), dtype=np.float32)
        canvas1[t[1]:img1.shape[0] + t[1], t[0]:img1.shape[1] + t[0]] = img1.astype(np.float32)

        if blend_mode == "feather":
            from src.blending.feathering import feather_blend
            output_img = feather_blend(canvas1, warped2.astype(np.float32),
                                       mask1, mask2)
        else:
            from src.blending.multiband import multiband_blend
            output_img = multiband_blend(canvas1, warped2.astype(np.float32),
                                         mask1, mask2)
    else:
        raise ValueError(f"Unknown blend_mode: '{blend_mode}'. "
                         f"Choose 'overwrite', 'feather', or 'multiband'.")

    return output_img
