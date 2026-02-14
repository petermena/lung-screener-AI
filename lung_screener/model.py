"""3D CNN architectures for lung nodule classification.

Provides ResNet3D and DenseNet3D models that classify 3D patches
as nodule vs. non-nodule, with optional malignancy scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D residual block with two conv layers and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NoduleResNet3D(nn.Module):
    """3D ResNet for nodule classification.

    Takes a 3D patch (e.g., 48x48x48) and outputs:
    - nodule probability (binary classification)
    - optional malignancy score (regression)

    Architecture:
        conv -> 4 residual stages -> global avg pool -> FC layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        predict_malignancy: bool = False,
    ):
        super().__init__()
        self.predict_malignancy = predict_malignancy

        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, base_filters, kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_filters)

        # Residual stages with increasing filters and spatial downsampling
        self.stage1 = self._make_stage(base_filters, base_filters, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(base_filters, base_filters * 2, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(base_filters * 2, base_filters * 4, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(base_filters * 4, base_filters * 8, num_blocks=2, stride=2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Optional malignancy regression head
        if predict_malignancy:
            self.malignancy_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(base_filters * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # Output in [0, 1], scale to [1, 5] externally
            )

        self._initialize_weights()

    def _make_stage(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 1, D, H, W).

        Returns:
            Dict with:
                - logits: (B, num_classes) classification logits
                - features: (B, 256) feature vector from GAP
                - malignancy: (B, 1) malignancy score (if enabled)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        features = self.gap(x).flatten(1)

        result = {
            "logits": self.classifier(features),
            "features": features,
        }

        if self.predict_malignancy:
            result["malignancy"] = self.malignancy_head(features)

        return result


class DenseBlock3D(nn.Module):
    """3D Dense block with multiple dense layers."""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))

    def _make_dense_layer(self, in_channels: int, growth_rate: int) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, growth_rate * 4, kernel_size=1, bias=False),
            nn.BatchNorm3d(growth_rate * 4),
            nn.ReLU(),
            nn.Conv3d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionBlock3D(nn.Module):
    """Transition block to reduce spatial dimensions and channel count."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NoduleDenseNet3D(nn.Module):
    """3D DenseNet for nodule classification.

    Alternative architecture with dense connectivity for better
    gradient flow and feature reuse.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        growth_rate: int = 16,
        block_layers: tuple[int, ...] = (4, 8, 16, 12),
        predict_malignancy: bool = False,
    ):
        super().__init__()
        self.predict_malignancy = predict_malignancy

        # Initial convolution
        num_features = growth_rate * 2
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm3d(num_features),
            nn.ReLU(),
        )

        # Dense blocks and transitions
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock3D(num_features, growth_rate, num_layers)
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionBlock3D(num_features, num_features // 2)
                self.blocks.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm3d(num_features)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )

        if predict_malignancy:
            self.malignancy_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.final_bn(x))
        features = self.gap(x).flatten(1)

        result = {
            "logits": self.classifier(features),
            "features": features,
        }

        if self.predict_malignancy:
            result["malignancy"] = self.malignancy_head(features)

        return result


def build_model(config: dict) -> nn.Module:
    """Factory function to build model from config.

    Args:
        config: Model configuration dict.

    Returns:
        Instantiated model.
    """
    model_config = config.get("model", {})
    arch = model_config.get("architecture", "resnet3d")
    in_channels = model_config.get("in_channels", 1)
    num_classes = model_config.get("num_classes", 2)

    if arch == "resnet3d":
        return NoduleResNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif arch == "densenet3d":
        return NoduleDenseNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use 'resnet3d' or 'densenet3d'.")
