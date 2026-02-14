"""Tests for the 3D CNN model architectures."""

import torch

from lung_screener.model import NoduleDenseNet3D, NoduleResNet3D, build_model


class TestNoduleResNet3D:
    def test_forward_shape(self):
        model = NoduleResNet3D(in_channels=1, num_classes=2)
        x = torch.randn(2, 1, 48, 48, 48)
        output = model(x)

        assert output["logits"].shape == (2, 2)
        assert output["features"].shape[0] == 2

    def test_forward_with_malignancy(self):
        model = NoduleResNet3D(in_channels=1, num_classes=2, predict_malignancy=True)
        x = torch.randn(2, 1, 48, 48, 48)
        output = model(x)

        assert "malignancy" in output
        assert output["malignancy"].shape == (2, 1)
        # Malignancy should be in [0, 1] (sigmoid output)
        assert output["malignancy"].min() >= 0
        assert output["malignancy"].max() <= 1

    def test_different_patch_sizes(self):
        model = NoduleResNet3D(in_channels=1, num_classes=2)
        for size in [32, 48, 64]:
            x = torch.randn(1, 1, size, size, size)
            output = model(x)
            assert output["logits"].shape == (1, 2)


class TestNoduleDenseNet3D:
    def test_forward_shape(self):
        model = NoduleDenseNet3D(in_channels=1, num_classes=2)
        x = torch.randn(2, 1, 48, 48, 48)
        output = model(x)

        assert output["logits"].shape == (2, 2)

    def test_forward_with_malignancy(self):
        model = NoduleDenseNet3D(in_channels=1, num_classes=2, predict_malignancy=True)
        x = torch.randn(1, 1, 48, 48, 48)
        output = model(x)

        assert "malignancy" in output
        assert output["malignancy"].shape == (1, 1)


class TestBuildModel:
    def test_build_resnet(self):
        config = {"model": {"architecture": "resnet3d", "in_channels": 1, "num_classes": 2}}
        model = build_model(config)
        assert isinstance(model, NoduleResNet3D)

    def test_build_densenet(self):
        config = {"model": {"architecture": "densenet3d", "in_channels": 1, "num_classes": 2}}
        model = build_model(config)
        assert isinstance(model, NoduleDenseNet3D)

    def test_build_unknown_raises(self):
        config = {"model": {"architecture": "unknown"}}
        try:
            build_model(config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
