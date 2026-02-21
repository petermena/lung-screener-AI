"""Tests for the 3D CNN model architectures."""

import torch

from lung_screener.model import (
    NoduleDenseNet3D,
    NoduleResNet3D,
    NoduleSEResNeXt3D,
    SEBlock3D,
    SEResNeXtBlock3D,
    build_model,
)


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


class TestSEBlock3D:
    def test_forward_preserves_shape(self):
        se = SEBlock3D(channels=64, reduction=16)
        x = torch.randn(2, 64, 12, 12, 12)
        out = se(x)
        assert out.shape == x.shape

    def test_output_range(self):
        """SE attention should scale features, not zero them all out."""
        se = SEBlock3D(channels=32, reduction=8)
        x = torch.ones(1, 32, 6, 6, 6)
        out = se(x)
        # After sigmoid attention, output should be in (0, 1) * input
        assert out.min() >= 0
        assert out.max() <= 1.0 + 1e-6  # sigmoid output <= 1


class TestSEResNeXtBlock3D:
    def test_forward_same_channels(self):
        block = SEResNeXtBlock3D(64, 64, stride=1, cardinality=16, bottleneck_width=4)
        x = torch.randn(2, 64, 12, 12, 12)
        out = block(x)
        assert out.shape == x.shape

    def test_forward_downsample(self):
        block = SEResNeXtBlock3D(64, 128, stride=2, cardinality=16, bottleneck_width=4)
        x = torch.randn(2, 64, 12, 12, 12)
        out = block(x)
        assert out.shape == (2, 128, 6, 6, 6)

    def test_stochastic_depth(self):
        block = SEResNeXtBlock3D(64, 64, drop_path_rate=0.999, cardinality=16, bottleneck_width=4)
        x = torch.randn(1, 64, 8, 8, 8)
        block.train()
        # With very high drop rate, output should sometimes equal identity
        out = block(x)
        assert out.shape == x.shape


class TestNoduleSEResNeXt3D:
    def test_forward_shape(self):
        model = NoduleSEResNeXt3D(in_channels=1, num_classes=2, base_filters=32, cardinality=16)
        x = torch.randn(2, 1, 48, 48, 48)
        output = model(x)

        assert output["logits"].shape == (2, 2)
        assert output["features"].shape[0] == 2
        # Multi-scale fusion: features should be s2_ch + s3_ch + s4_ch
        assert output["features"].shape[1] == 32 * 2 + 32 * 4 + 32 * 8  # 448

    def test_forward_with_malignancy(self):
        model = NoduleSEResNeXt3D(
            in_channels=1, num_classes=2, base_filters=32,
            cardinality=16, predict_malignancy=True,
        )
        x = torch.randn(1, 1, 48, 48, 48)
        output = model(x)

        assert "malignancy" in output
        assert output["malignancy"].shape == (1, 1)
        assert output["malignancy"].min() >= 0
        assert output["malignancy"].max() <= 1

    def test_forward_with_nodule_type(self):
        model = NoduleSEResNeXt3D(
            in_channels=1, num_classes=2, base_filters=32,
            cardinality=16, predict_nodule_type=True,
        )
        x = torch.randn(1, 1, 48, 48, 48)
        output = model(x)

        assert "nodule_type_logits" in output
        assert output["nodule_type_logits"].shape == (1, 3)

    def test_different_patch_sizes(self):
        model = NoduleSEResNeXt3D(in_channels=1, num_classes=2, base_filters=32, cardinality=16)
        for size in [32, 48, 64]:
            x = torch.randn(1, 1, size, size, size)
            output = model(x)
            assert output["logits"].shape == (1, 2)

    def test_parameter_count_reasonable(self):
        """SE-ResNeXt3D should have more params than basic ResNet but not absurd."""
        model = NoduleSEResNeXt3D(in_channels=1, num_classes=2, base_filters=64, cardinality=32)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be between 1M and 50M parameters
        assert 1_000_000 < n_params < 50_000_000


class TestBuildModel:
    def test_build_resnet(self):
        config = {"model": {"architecture": "resnet3d", "in_channels": 1, "num_classes": 2}}
        model = build_model(config)
        assert isinstance(model, NoduleResNet3D)

    def test_build_densenet(self):
        config = {"model": {"architecture": "densenet3d", "in_channels": 1, "num_classes": 2}}
        model = build_model(config)
        assert isinstance(model, NoduleDenseNet3D)

    def test_build_se_resnext(self):
        config = {
            "model": {
                "architecture": "se_resnext3d",
                "in_channels": 1,
                "num_classes": 2,
                "base_filters": 32,
                "cardinality": 16,
                "bottleneck_width": 4,
                "se_reduction": 16,
                "drop_path_rate": 0.1,
            }
        }
        model = build_model(config)
        assert isinstance(model, NoduleSEResNeXt3D)

    def test_build_unknown_raises(self):
        config = {"model": {"architecture": "unknown"}}
        try:
            build_model(config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
