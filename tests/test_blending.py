"""
tests/test_blending.py
======================
Unit tests for src/blending/feathering.py and src/blending/multiband.py
on synthetic overlapping patches.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.blending.feathering import feather_blend
from src.blending.multiband  import multiband_blend


# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------

def _make_patches(h=200, w=300, overlap_frac=0.4):
    """
    Create two overlapping colour patches on a common canvas.

    Returns
    -------
    canvas1 : float32 (h, w, 3)  – left image (red tint) on canvas
    canvas2 : float32 (h, w, 3)  – right image (blue tint) on canvas
    mask1   : uint8 (h, w)       – 255 where canvas1 is filled
    mask2   : uint8 (h, w)       – 255 where canvas2 is filled
    """
    overlap = int(w * overlap_frac)
    mid     = w // 2

    c1 = np.zeros((h, w, 3), dtype=np.float32)
    c1[:, :mid + overlap // 2] = [0, 0, 200]   # red patch (BGR)

    c2 = np.zeros((h, w, 3), dtype=np.float32)
    c2[:, mid - overlap // 2:] = [200, 0, 0]   # blue patch (BGR)

    m1 = np.zeros((h, w), dtype=np.uint8)
    m1[:, :mid + overlap // 2] = 255

    m2 = np.zeros((h, w), dtype=np.uint8)
    m2[:, mid - overlap // 2:] = 255

    return c1, c2, m1, m2


# ---------------------------------------------------------------------------
# Feathering tests
# ---------------------------------------------------------------------------

class TestFeatherBlend:
    def test_output_shape(self):
        c1, c2, m1, m2 = _make_patches()
        result = feather_blend(c1, c2, m1, m2)
        assert result.shape == c1.shape
        assert result.dtype == np.uint8

    def test_output_in_valid_range(self):
        c1, c2, m1, m2 = _make_patches()
        result = feather_blend(c1, c2, m1, m2)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_non_overlap_regions_preserved(self):
        """Pixels covered only by canvas1 should equal canvas1."""
        c1, c2, m1, m2 = _make_patches()
        result = feather_blend(c1, c2, m1, m2)
        only1 = (m1 > 0) & (m2 == 0)
        if only1.any():
            np.testing.assert_array_equal(
                result[only1],
                np.clip(c1[only1], 0, 255).astype(np.uint8),
                err_msg="Non-overlapping pixels of canvas1 should be unchanged"
            )

    def test_overlap_is_blended(self):
        """Overlap pixels should NOT equal either canvas alone (they're blended)."""
        c1, c2, m1, m2 = _make_patches()
        result = feather_blend(c1, c2, m1, m2)
        both = (m1 > 0) & (m2 > 0)
        if both.any():
            # Result in overlap should differ from pure c1 (because c2 contributes)
            assert not np.array_equal(
                result[both].astype(np.float32),
                np.clip(c1[both], 0, 255)
            ), "Overlap should be a blend, not identical to canvas1"


# ---------------------------------------------------------------------------
# Multi-band blending tests
# ---------------------------------------------------------------------------

class TestMultibandBlend:
    def test_output_shape(self):
        c1, c2, m1, m2 = _make_patches()
        result = multiband_blend(c1, c2, m1, m2, levels=3)
        assert result.shape == c1.shape
        assert result.dtype == np.uint8

    def test_output_in_valid_range(self):
        c1, c2, m1, m2 = _make_patches()
        result = multiband_blend(c1, c2, m1, m2, levels=3)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_varying_pyramid_levels(self):
        c1, c2, m1, m2 = _make_patches(h=256, w=512)
        for lvl in (1, 3, 5):
            result = multiband_blend(c1, c2, m1, m2, levels=lvl)
            assert result.shape == c1.shape, f"Shape mismatch at levels={lvl}"

    def test_seam_smoother_than_overwrite(self):
        """
        Feathered/pyramid blend should have lower gradient at seam than hard overwrite.
        """
        import cv2
        c1, c2, m1, m2 = _make_patches(h=200, w=300, overlap_frac=0.3)
        both = (m1 > 0) & (m2 > 0)
        seam_col = np.where(both.any(axis=0))[0]
        if len(seam_col) < 2:
            pytest.skip("Not enough overlap for seam test")
        mid_x = int(seam_col.mean())

        # Hard overwrite: paste c1 on top of c2
        overwrite = c2.copy().astype(np.uint8)
        overwrite[m1 > 0] = np.clip(c1[m1 > 0], 0, 255).astype(np.uint8)

        # Pyramid blend
        blended = multiband_blend(c1, c2, m1, m2, levels=3)

        # In overwrite, the abrupt boundary is at seam_col[-1] where m1 ends
        def peak_grad_in_overlap(img, cols):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            col_grad = np.abs(np.diff(gray[:, cols[0]:cols[-1] + 2], axis=1))
            return col_grad.mean(axis=0).max()

        grad_overwrite = peak_grad_in_overlap(overwrite, seam_col)
        grad_blended   = peak_grad_in_overlap(blended,   seam_col)
        assert grad_blended < grad_overwrite, \
            f"Blended peak seam gradient ({grad_blended:.2f}) should be < overwrite ({grad_overwrite:.2f})"
