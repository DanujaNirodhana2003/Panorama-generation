"""
src/matching/matcher.py
=======================
Feature matching module — FLANN (for SIFT) and BruteForce (for ORB).

M3 Baseline reference
---------------------
FLANN KD-Tree (5 trees, 50 checks) + Lowe's ratio test (threshold 0.70)
produced 76.56 % inlier ratio after RANSAC on the test image pair.
"""

import cv2
import numpy as np


def flann_match(des1: np.ndarray, des2: np.ndarray,
                ratio_thresh: float = 0.70,
                trees: int = 5,
                checks: int = 50):
    """
    Match float descriptors (e.g. SIFT) using FLANN KD-Tree + Lowe's ratio test.

    Parameters
    ----------
    des1, des2     : ndarray  – float32 descriptors from extractor.extract_sift()
    ratio_thresh   : float    – Lowe's ratio threshold (default 0.70)
    trees          : int      – number of KD-Tree trees (default 5)
    checks         : int      – search checks (default 50)

    Returns
    -------
    good_matches : list[cv2.DMatch]  – matches that pass the ratio test
    all_matches  : list[(cv2.DMatch, cv2.DMatch)]  – raw kNN pairs
    """
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    all_matches  = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in all_matches if m.distance < ratio_thresh * n.distance]
    return good_matches, all_matches


def brute_force_match(des1: np.ndarray, des2: np.ndarray,
                      norm_type=cv2.NORM_HAMMING,
                      top_k: int = 200):
    """
    Match binary descriptors (e.g. ORB) using BruteForce + Hamming distance.

    Parameters
    ----------
    des1, des2  : ndarray   – uint8 ORB descriptors
    norm_type   : int       – cv2.NORM_HAMMING or cv2.NORM_L2
    top_k       : int       – keep the best top_k matches (sorted by distance)

    Returns
    -------
    matches : list[cv2.DMatch]
    """
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:top_k]


def extract_point_pairs(kp1, kp2, good_matches):
    """
    Convert matched keypoints to float32 point arrays suitable for findHomography.

    Returns
    -------
    src_pts : ndarray, shape (N, 1, 2)
    dst_pts : ndarray, shape (N, 1, 2)
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts
