"""False positive reduction via second-stage classification.

Applies a separate, more focused classifier to candidates that passed
the initial detection threshold, reducing false positives while
maintaining high sensitivity.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class FPReductionNet(nn.Module):
    """Second-stage 3D CNN for false positive reduction.

    A deeper, more discriminative classifier that operates on candidates
    already flagged by the first-stage detector. Uses a larger receptive
    field and multi-scale feature analysis.

    Architecture:
        Multi-scale conv -> residual blocks -> attention -> classifier
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()

        # Multi-scale input processing
        self.scale1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters // 2),
            nn.ReLU(),
        )
        self.scale2 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(base_filters // 2),
            nn.ReLU(),
        )

        # Main residual path
        self.block1 = self._res_block(base_filters, base_filters, stride=1)
        self.block2 = self._res_block(base_filters, base_filters * 2, stride=2)
        self.block3 = self._res_block(base_filters * 2, base_filters * 4, stride=2)
        self.block4 = self._res_block(base_filters * 4, base_filters * 8, stride=2)

        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_filters * 8, base_filters * 2),
            nn.ReLU(),
            nn.Linear(base_filters * 2, base_filters * 8),
            nn.Sigmoid(),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

        # Classifier: outputs probability of being a true nodule
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        self._initialize_weights()

    def _res_block(self, in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
        return _ResBlock(in_ch, out_ch, stride)

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
        # Multi-scale feature extraction
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        x = torch.cat([f1, f2], dim=1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Channel attention
        attn = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x * attn

        features = self.gap(x).flatten(1)
        logits = self.classifier(features)

        return {"logits": logits, "features": features}


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class FalsePositiveReducer:
    """Applies a second-stage classifier to reduce false positives.

    Loads a separately trained FP-reduction model and re-scores
    candidates from the first-stage detector. Only candidates that
    pass both stages are reported.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold
        self.model = FPReductionNet().to(self.device)

        if model_path:
            self._load_model(model_path)

        self.model.eval()

    def _load_model(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info(f"Loaded FP reduction model from {path}")

    @torch.no_grad()
    def filter_findings(
        self,
        patches: np.ndarray,
        findings: list,
        batch_size: int = 32,
    ) -> list:
        """Re-score findings with the FP reduction model.

        Args:
            patches: Array of shape (N, 1, D, H, W) for each finding.
            findings: List of NoduleFinding objects to filter.
            batch_size: Batch size for inference.

        Returns:
            Filtered list of NoduleFinding objects that passed FP reduction.
        """
        if len(findings) == 0 or patches.shape[0] == 0:
            return findings

        all_probs = []
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(
                patches[i : i + batch_size]
            ).float().to(self.device)

            with autocast():
                output = self.model(batch)

            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

        # Filter findings that pass the FP reduction threshold
        filtered = []
        for finding, prob in zip(findings, all_probs):
            if prob >= self.threshold:
                # Average the two-stage confidences
                finding.confidence = (finding.confidence + float(prob)) / 2.0
                filtered.append(finding)
            else:
                logger.debug(
                    f"FP reduction filtered out finding at "
                    f"({finding.x:.1f}, {finding.y:.1f}, {finding.z:.1f}) "
                    f"with FP score {prob:.3f}"
                )

        logger.info(
            f"FP reduction: {len(findings)} -> {len(filtered)} findings "
            f"(removed {len(findings) - len(filtered)} false positives)"
        )
        return filtered
