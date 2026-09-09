"""
src/features/extractor.py
=========================
Feature extraction module — SIFT and ORB detectors.

M3 Baseline reference
---------------------
SIFT + FLANN pipeline produces:
  - Keypoints: ~1200 (img1), ~1100 (img2) on the test pair
  - Inlier ratio : 76.56 %
  - RMSE         : 0.54 px
  - Speed        : ~0.65 s/pair (feature extraction is ~40% of that)
"""

import cv2
import numpy as np


def extract_sift(image: np.ndarray, gray: bool = True):
    """
    Detect SIFT keypoints and compute float descriptors.

    Parameters
    ----------
    image : ndarray  – BGR (or grayscale) image
    gray  : bool     – if True, convert BGR→Gray before detection

    Returns
    -------
    keypoints   : list[cv2.KeyPoint]
    descriptors : ndarray, shape (N, 128), float32
    """
    sift = cv2.SIFT_create()
    if gray and image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    kp, des = sift.detectAndCompute(img_gray, None)
    return kp, des


def extract_orb(image: np.ndarray, n_features: int = 2000, gray: bool = True):
    """
    Detect ORB keypoints and compute binary descriptors.

    Parameters
    ----------
    image      : ndarray  – BGR (or grayscale) image
    n_features : int      – max keypoints to retain
    gray       : bool     – if True, convert BGR→Gray before detection

    Returns
    -------
    keypoints   : list[cv2.KeyPoint]
    descriptors : ndarray, shape (N, 32), uint8
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    if gray and image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    kp, des = orb.detectAndCompute(img_gray, None)
    return kp, des
