"""Tests for the preprocessing pipeline."""

import numpy as np

from lung_screener.preprocessing import apply_hu_window, extract_patch


class TestApplyHUWindow:
    def test_clipping(self):
        volume = np.array([-2000.0, -1200.0, 0.0, 600.0, 2000.0], dtype=np.float32)
        result = apply_hu_window(volume, hu_min=-1200, hu_max=600, normalize=False)

        assert result[0] == -1200.0
        assert result[-1] == 600.0

    def test_normalization(self):
        volume = np.array([-1200.0, -300.0, 600.0], dtype=np.float32)
        result = apply_hu_window(volume, hu_min=-1200, hu_max=600, normalize=True)

        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[-1] - 1.0) < 1e-6
        assert 0.0 <= result[1] <= 1.0

    def test_output_dtype(self):
        volume = np.array([0.0, 100.0], dtype=np.float64)
        result = apply_hu_window(volume)
        assert result.dtype == np.float32


class TestExtractPatch:
    def test_center_patch(self):
        volume = np.random.rand(100, 100, 100).astype(np.float32)
        patch = extract_patch(volume, (50, 50, 50), (48, 48, 48))
        assert patch.shape == (48, 48, 48)

    def test_boundary_patch_padded(self):
        volume = np.ones((100, 100, 100), dtype=np.float32)
        # Center near edge - should zero-pad
        patch = extract_patch(volume, (0, 0, 0), (48, 48, 48))
        assert patch.shape == (48, 48, 48)
        # Some region should be zero (padding)
        assert patch.sum() < 48 * 48 * 48

    def test_single_voxel_center(self):
        volume = np.zeros((100, 100, 100), dtype=np.float32)
        volume[50, 50, 50] = 1.0
        patch = extract_patch(volume, (50, 50, 50), (48, 48, 48))
        assert patch[24, 24, 24] == 1.0
