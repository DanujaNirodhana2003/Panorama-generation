"""
tests/test_features.py
======================
Unit tests for src/features/extractor.py
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.extractor import extract_sift, extract_orb


def _synthetic_image(rich: bool = True) -> np.ndarray:
    """Create a small BGR image with enough texture for feature detection."""
    import cv2
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (180, 180, 180)
    if rich:
        cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 200), -1)
        cv2.circle   (img, (200, 100), 60, (0, 200, 0), -1)
        cv2.putText  (img, "TEST", (100, 160),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)
    return img


class TestExtractSift:
    def test_returns_keypoints_and_descriptors(self):
        img = _synthetic_image()
        kp, des = extract_sift(img)
        assert len(kp) > 0, "SIFT should detect keypoints on a textured image"
        assert des is not None
        assert des.shape[1] == 128, "SIFT descriptor length must be 128"
        assert des.dtype == np.float32

    def test_grayscale_input(self):
        import cv2
        img = _synthetic_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = extract_sift(gray, gray=False)
        assert len(kp) > 0

    def test_returns_empty_on_blank_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        kp, des = extract_sift(img)
        # Blank image may have 0 keypoints — just ensure no crash
        assert des is None or des.ndim == 2


class TestExtractOrb:
    def test_returns_keypoints_and_descriptors(self):
        img = _synthetic_image()
        kp, des = extract_orb(img)
        assert len(kp) > 0, "ORB should detect keypoints on a textured image"
        assert des is not None
        assert des.shape[1] == 32, "ORB descriptor length must be 32"
        assert des.dtype == np.uint8

    def test_n_features_respected(self):
        img = _synthetic_image()
        kp, des = extract_orb(img, n_features=5)
        assert len(kp) <= 5
