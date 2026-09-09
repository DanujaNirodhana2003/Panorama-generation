"""
tests/test_matching.py
======================
Unit tests for src/matching/matcher.py — ratio-test filtering and point-pair extraction.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.matching.matcher import flann_match, brute_force_match, extract_point_pairs
from src.features.extractor import extract_sift, extract_orb


def _two_synthetic_overlapping():
    """Return two BGR images with a known 200-px overlap."""
    import cv2
    canvas = np.zeros((300, 600, 3), dtype=np.uint8)
    canvas[:] = (150, 150, 150)
    cv2.rectangle(canvas, (50,  50), (200, 250), (0, 0, 255), -1)
    cv2.circle   (canvas, (350, 150), 80,  (0, 200, 0),  -1)
    cv2.rectangle(canvas, (450, 50), (560, 250), (200, 0, 0), -1)
    img1 = canvas[:, :400].copy()
    img2 = canvas[:, 200:].copy()
    return img1, img2


class TestFlannMatch:
    def test_returns_fewer_matches_with_stricter_ratio(self):
        img1, img2 = _two_synthetic_overlapping()
        kp1, des1 = extract_sift(img1)
        kp2, des2 = extract_sift(img2)
        if des1 is None or des2 is None:
            pytest.skip("No descriptors on synthetic image")

        good_loose, _ = flann_match(des1, des2, ratio_thresh=0.9)
        good_strict, _ = flann_match(des1, des2, ratio_thresh=0.5)
        assert len(good_strict) <= len(good_loose), \
            "Stricter ratio threshold must yield fewer or equal matches"

    def test_good_matches_pass_ratio(self):
        """Manually verify that every returned match satisfies the ratio test."""
        img1, img2 = _two_synthetic_overlapping()
        kp1, des1 = extract_sift(img1)
        kp2, des2 = extract_sift(img2)
        if des1 is None or des2 is None:
            pytest.skip("No descriptors")

        ratio = 0.7
        good, all_m = flann_match(des1, des2, ratio_thresh=ratio)
        for m, n in all_m:
            if m in good:
                assert m.distance < ratio * n.distance, \
                    "Match in good_matches violated ratio test"


class TestExtractPointPairs:
    def test_shape(self):
        img1, img2 = _two_synthetic_overlapping()
        kp1, des1 = extract_sift(img1)
        kp2, des2 = extract_sift(img2)
        if des1 is None or des2 is None:
            pytest.skip("No descriptors")
        good, _ = flann_match(des1, des2, ratio_thresh=0.75)
        if not good:
            pytest.skip("No good matches on synthetic images")
        src, dst = extract_point_pairs(kp1, kp2, good)
        assert src.shape == (len(good), 1, 2)
        assert dst.shape == (len(good), 1, 2)
        assert src.dtype == np.float32


class TestBruteForceMatch:
    def test_orb_brute_force(self):
        img1, img2 = _two_synthetic_overlapping()
        kp1, des1 = extract_orb(img1)
        kp2, des2 = extract_orb(img2)
        if des1 is None or des2 is None:
            pytest.skip("No ORB descriptors")
        matches = brute_force_match(des1, des2, top_k=50)
        assert len(matches) <= 50
        # Matches sorted by distance (ascending)
        if len(matches) > 1:
            assert matches[0].distance <= matches[-1].distance
