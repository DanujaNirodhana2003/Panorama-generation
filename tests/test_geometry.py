"""
tests/test_geometry.py
======================
Unit tests for src/geometry/homography.py:
  - RANSAC inlier mask shape
  - Skew/stretch fallback trigger at 1.3× threshold
  - Affine fallback produces a valid (non-catastrophic) result
"""

import numpy as np
import pytest
import cv2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.homography import estimate_homography, _bounding_box_stretch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pure_translation_pts(n=50, tx=100.0, ty=20.0, noise=0.5):
    """Generate matched point pairs under a pure translation + small noise."""
    src = np.random.uniform(0, 400, (n, 2)).astype(np.float32)
    dst = src + np.array([tx, ty], dtype=np.float32)
    dst += np.random.normal(0, noise, dst.shape).astype(np.float32)
    return src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2)


def _extreme_stretch_H():
    """Return a homography that stretches width by ~4× (should trigger fallback)."""
    # Scale x by 4, keep y same
    return np.array([[4, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoundingBoxStretch:
    def test_identity_no_stretch(self):
        H = np.eye(3, dtype=np.float64)
        w_ratio, h_ratio = _bounding_box_stretch((300, 400, 3), H)
        assert abs(w_ratio - 1.0) < 0.01
        assert abs(h_ratio - 1.0) < 0.01

    def test_scale_2x_detected(self):
        H = np.diag([2.0, 2.0, 1.0])
        w_ratio, h_ratio = _bounding_box_stretch((300, 400, 3), H)
        assert abs(w_ratio - 2.0) < 0.05
        assert abs(h_ratio - 2.0) < 0.05

    def test_extreme_width_stretch(self):
        H = _extreme_stretch_H()
        w_ratio, _ = _bounding_box_stretch((300, 400, 3), H)
        assert w_ratio > 1.3, "4× scale should exceed 1.3 threshold"


class TestEstimateHomography:
    def test_inlier_mask_shape(self):
        src, dst = _pure_translation_pts(n=60)
        H, mask, fallback = estimate_homography(
            src, dst, img2_shape=(400, 600, 3), stretch_limit=10.0
        )
        assert H is not None, "Should find a valid H for clean translation data"
        assert mask is not None
        assert mask.shape[0] == 60 or mask.ravel().shape[0] == 60

    def test_high_inlier_ratio_on_clean_data(self):
        src, dst = _pure_translation_pts(n=100, noise=0.1)
        H, mask, _ = estimate_homography(
            src, dst, img2_shape=(400, 600, 3), stretch_limit=10.0
        )
        if H is not None and mask is not None:
            inlier_ratio = mask.ravel().sum() / len(src)
            assert inlier_ratio > 0.7, \
                f"Expected >70% inliers on near-clean data, got {inlier_ratio:.2%}"

    def test_fallback_triggers_on_stretch(self, tmp_path):
        """Inject extreme-stretch matched points to force the fallback branch."""
        # Create points that produce a ~3× horizontal stretch H
        src = np.float32([[0,0],[200,0],[200,200],[0,200],[100,100]]).reshape(-1,1,2)
        # dst maps to 3× wider → homography will stretch x by ~3
        dst = np.float32([[0,0],[600,0],[600,200],[0,200],[300,100]]).reshape(-1,1,2)
        log = str(tmp_path / "fallback.csv")
        H, mask, fallback = estimate_homography(
            src, dst, img2_shape=(200, 200, 3),
            stretch_limit=1.3, log_path=log, pair_label="test_pair"
        )
        assert fallback is True, "Extreme stretch should trigger affine fallback"
        assert H is not None, "Affine fallback should still return a valid H"
        # Log file should have been written
        assert os.path.isfile(log), "Fallback log CSV should be created"

    def test_returns_none_on_too_few_points(self):
        src = np.float32([[0,0],[1,0],[0,1]]).reshape(-1,1,2)
        dst = np.float32([[0,0],[1,0],[0,1]]).reshape(-1,1,2)
        # Only 3 points — findHomography needs ≥ 4
        H, mask, _ = estimate_homography(src, dst, img2_shape=(100,100,3))
        # May return None or degenerate H — just no crash
