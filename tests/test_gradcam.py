"""Tests for GradCAM 3D saliency map generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lung_screener.gradcam import GradCAM3D, render_slices, render_three_plane
from lung_screener.model import NoduleDenseNet3D, NoduleResNet3D


@pytest.fixture
def resnet_model():
    model = NoduleResNet3D(in_channels=1, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def densenet_model():
    model = NoduleDenseNet3D(in_channels=1, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def dummy_patch():
    """48x48x48 patch with a bright blob in the center (simulated nodule)."""
    patch = np.random.rand(48, 48, 48).astype(np.float32) * 0.2
    # Add a spherical bright region in the center
    z, y, x = np.ogrid[-24:24, -24:24, -24:24]
    mask = (z**2 + y**2 + x**2) < 8**2
    patch[mask] = 0.8
    return patch


class TestGradCAM3DResNet:
    def test_auto_detects_target_layer(self, resnet_model):
        gc = GradCAM3D(resnet_model)
        assert gc.target_layer is resnet_model.stage4
        gc.release()

    def test_heatmap_shape_matches_input(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        heatmap, pred = gc.generate(tensor)

        assert heatmap.shape == (48, 48, 48)
        gc.release()

    def test_heatmap_values_normalized(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        heatmap, _ = gc.generate(tensor)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        gc.release()

    def test_prediction_dict_fields(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        _, pred = gc.generate(tensor)

        assert "class" in pred
        assert "confidence" in pred
        assert "logits" in pred
        assert "target_class" in pred
        assert "nodule_probability" in pred
        assert pred["class"] in (0, 1)
        assert 0.0 <= pred["confidence"] <= 1.0
        assert 0.0 <= pred["nodule_probability"] <= 1.0
        gc.release()

    def test_target_class_override(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        heatmap_0, pred_0 = gc.generate(tensor, target_class=0)
        heatmap_1, pred_1 = gc.generate(tensor, target_class=1)

        assert pred_0["target_class"] == 0
        assert pred_1["target_class"] == 1
        # Heatmaps for different target classes should generally differ
        gc.release()

    def test_4d_input_auto_unsqueeze(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0)  # (1, D, H, W) — no batch dim
        heatmap, _ = gc.generate(tensor)
        assert heatmap.shape == (48, 48, 48)
        gc.release()

    def test_explicit_target_layer(self, resnet_model, dummy_patch):
        gc = GradCAM3D(resnet_model, target_layer=resnet_model.stage3)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        heatmap, _ = gc.generate(tensor)
        assert heatmap.shape == (48, 48, 48)
        gc.release()


class TestGradCAM3DDenseNet:
    def test_auto_detects_target_layer(self, densenet_model):
        gc = GradCAM3D(densenet_model)
        assert gc.target_layer is densenet_model.blocks[-1]
        gc.release()

    def test_heatmap_shape(self, densenet_model, dummy_patch):
        gc = GradCAM3D(densenet_model)
        tensor = torch.from_numpy(dummy_patch).float().unsqueeze(0).unsqueeze(0)
        heatmap, pred = gc.generate(tensor)
        assert heatmap.shape == (48, 48, 48)
        assert 0.0 <= pred["confidence"] <= 1.0
        gc.release()


class TestVisualization:
    def test_render_slices_creates_file(self, dummy_patch):
        heatmap = np.random.rand(48, 48, 48).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_slices(
                dummy_patch, heatmap,
                Path(tmpdir) / "test_slices.png",
                num_slices=6,
                prediction={"class": 1, "confidence": 0.95},
            )
            assert path.exists()
            assert path.stat().st_size > 0

    def test_render_three_plane_creates_file(self, dummy_patch):
        heatmap = np.random.rand(48, 48, 48).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_three_plane(
                dummy_patch, heatmap,
                Path(tmpdir) / "test_3plane.png",
                prediction={"class": 0, "confidence": 0.72},
            )
            assert path.exists()
            assert path.stat().st_size > 0

    def test_render_slices_no_prediction(self, dummy_patch):
        heatmap = np.random.rand(48, 48, 48).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_slices(
                dummy_patch, heatmap,
                Path(tmpdir) / "no_pred.png",
            )
            assert path.exists()

    def test_render_creates_parent_dirs(self, dummy_patch):
        heatmap = np.random.rand(48, 48, 48).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_slices(
                dummy_patch, heatmap,
                Path(tmpdir) / "sub" / "dir" / "test.png",
            )
            assert path.exists()
