"""3D CNN architectures for lung nodule classification.

Provides ResNet3D, DenseNet3D, and SE-ResNeXt3D models that classify
3D patches as nodule vs. non-nodule, with optional malignancy scoring
and nodule type classification (solid, part-solid, ground-glass).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Nodule type labels
NODULE_TYPES = ["solid", "part_solid", "ground_glass"]


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


# ======================================================================
# Squeeze-and-Excitation ResNeXt 3D
# ======================================================================


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps.

    Learns per-channel attention weights by squeezing spatial dimensions
    with global average pooling, then exciting via a two-layer FC
    bottleneck.  Proven to boost classification accuracy with negligible
    parameter overhead.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # Squeeze: global average pool over spatial dims
        s = x.view(b, c, -1).mean(dim=2)
        # Excitation
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1, 1)


class SEResNeXtBlock3D(nn.Module):
    """3D ResNeXt block with grouped convolutions + SE attention.

    Uses cardinality (number of groups) to increase representational
    power while keeping parameter count similar to standard ResNet.
    The SE block adds channel-wise recalibration after the residual
    mapping.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        stride: Spatial stride (use 2 for downsampling).
        cardinality: Number of groups for grouped convolution.
        bottleneck_width: Width per cardinality group.
        se_reduction: Squeeze-and-Excitation reduction ratio.
        drop_path_rate: Stochastic depth drop probability (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        cardinality: int = 32,
        bottleneck_width: int = 4,
        se_reduction: int = 16,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv3d(in_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(group_width)
        self.conv2 = nn.Conv3d(
            group_width, group_width, kernel_size=3,
            stride=stride, padding=1, groups=cardinality, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(group_width)
        self.conv3 = nn.Conv3d(group_width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels, se_reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        # Stochastic depth
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1, device=out.device).item() < self.drop_path_rate:
                return identity

        out += identity
        return F.relu(out)


class NoduleResNet3D(nn.Module):
    """3D ResNet for nodule classification.

    Takes a 3D patch (e.g., 48x48x48) and outputs:
    - nodule probability (binary classification)
    - optional malignancy score (regression)
    - optional nodule type (solid, part-solid, ground-glass)

    Architecture:
        conv -> 4 residual stages -> global avg pool -> FC layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        predict_malignancy: bool = False,
        predict_nodule_type: bool = False,
    ):
        super().__init__()
        self.predict_malignancy = predict_malignancy
        self.predict_nodule_type = predict_nodule_type

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

        # Optional nodule type classification head
        # 3 classes: solid (0), part_solid (1), ground_glass (2)
        if predict_nodule_type:
            self.nodule_type_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(base_filters * 8, 64),
                nn.ReLU(),
                nn.Linear(64, len(NODULE_TYPES)),
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
                - nodule_type_logits: (B, 3) nodule type logits (if enabled)
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

        if self.predict_nodule_type:
            result["nodule_type_logits"] = self.nodule_type_head(features)

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
        predict_nodule_type: bool = False,
    ):
        super().__init__()
        self.predict_malignancy = predict_malignancy
        self.predict_nodule_type = predict_nodule_type

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

        if predict_nodule_type:
            self.nodule_type_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, len(NODULE_TYPES)),
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

        if self.predict_nodule_type:
            result["nodule_type_logits"] = self.nodule_type_head(features)

        return result


class NoduleSEResNeXt3D(nn.Module):
    """3D SE-ResNeXt with multi-scale feature fusion for nodule classification.

    Combines three proven techniques for state-of-the-art performance:

    1. **ResNeXt grouped convolutions** — increased cardinality captures
       richer feature representations than standard ResNet at equal depth.
    2. **Squeeze-and-Excitation** — channel-wise attention recalibrates
       features adaptively, boosting informative channels.
    3. **Multi-scale feature fusion** — features from stages 2-4 are
       pooled and concatenated before the classifier, providing both
       fine-grained detail and global context.
    4. **Stochastic depth** — regularisation that randomly drops residual
       blocks during training for better generalisation.

    Architecture::

        stem_conv -> stage1 -> stage2 -> stage3 -> stage4
                                 |         |         |
                              pool+fc   pool+fc   pool+fc
                                 \\________||________/
                                    concat -> classifier

    Args:
        in_channels: Input channels (1 for CT grayscale).
        num_classes: Classification output classes.
        base_filters: Base channel width (doubled per stage).
        cardinality: Number of groups in grouped convolution.
        bottleneck_width: Width per group.
        se_reduction: SE bottleneck reduction ratio.
        drop_path_rate: Max stochastic depth rate (linearly scaled).
        predict_malignancy: Add malignancy regression head.
        predict_nodule_type: Add nodule type classification head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        cardinality: int = 32,
        bottleneck_width: int = 4,
        se_reduction: int = 16,
        drop_path_rate: float = 0.2,
        predict_malignancy: bool = False,
        predict_nodule_type: bool = False,
    ):
        super().__init__()
        self.predict_malignancy = predict_malignancy
        self.predict_nodule_type = predict_nodule_type

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Stage configurations: (out_channels, num_blocks, stride)
        stage_cfgs = [
            (base_filters, 2, 1),          # stage1
            (base_filters * 2, 3, 2),      # stage2
            (base_filters * 4, 3, 2),      # stage3
            (base_filters * 8, 2, 2),      # stage4
        ]

        # Count total blocks for linear stochastic depth schedule
        total_blocks = sum(n for _, n, _ in stage_cfgs)
        block_idx = 0

        stages = []
        ch_in = base_filters
        for ch_out, n_blocks, stride in stage_cfgs:
            blocks = []
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                blocks.append(SEResNeXtBlock3D(
                    ch_in if i == 0 else ch_out, ch_out,
                    stride=s,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    se_reduction=se_reduction,
                    drop_path_rate=dp,
                ))
                block_idx += 1
                ch_in = ch_out
            stages.append(nn.Sequential(*blocks))

        self.stage1, self.stage2, self.stage3, self.stage4 = stages

        # Multi-scale feature fusion: pool features from stages 2, 3, 4
        # and concatenate for a richer representation.
        s2_ch = stage_cfgs[1][0]  # base_filters * 2
        s3_ch = stage_cfgs[2][0]  # base_filters * 4
        s4_ch = stage_cfgs[3][0]  # base_filters * 8

        self.pool2 = nn.AdaptiveAvgPool3d(1)
        self.pool3 = nn.AdaptiveAvgPool3d(1)
        self.pool4 = nn.AdaptiveAvgPool3d(1)

        fused_ch = s2_ch + s3_ch + s4_ch  # sum of multi-scale channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if predict_malignancy:
            self.malignancy_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fused_ch, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        if predict_nodule_type:
            self.nodule_type_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(fused_ch, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, len(NODULE_TYPES)),
            )

        self._initialize_weights()

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
        x = self.stem(x)
        x = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        # Multi-scale fusion
        p2 = self.pool2(f2).flatten(1)
        p3 = self.pool3(f3).flatten(1)
        p4 = self.pool4(f4).flatten(1)
        features = torch.cat([p2, p3, p4], dim=1)

        result = {
            "logits": self.classifier(features),
            "features": features,
        }

        if self.predict_malignancy:
            result["malignancy"] = self.malignancy_head(features)

        if self.predict_nodule_type:
            result["nodule_type_logits"] = self.nodule_type_head(features)

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
    predict_nodule_type = model_config.get("predict_nodule_type", False)

    if arch == "resnet3d":
        return NoduleResNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            predict_nodule_type=predict_nodule_type,
        )
    elif arch == "densenet3d":
        return NoduleDenseNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            predict_nodule_type=predict_nodule_type,
        )
    elif arch == "se_resnext3d":
        return NoduleSEResNeXt3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=model_config.get("base_filters", 64),
            cardinality=model_config.get("cardinality", 32),
            bottleneck_width=model_config.get("bottleneck_width", 4),
            se_reduction=model_config.get("se_reduction", 16),
            drop_path_rate=model_config.get("drop_path_rate", 0.2),
            predict_nodule_type=predict_nodule_type,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. "
            "Use 'resnet3d', 'densenet3d', or 'se_resnext3d'."
        )
