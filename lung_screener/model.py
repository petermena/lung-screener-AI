from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from monai.networks.nets import DenseNet121, resnet18


ModelBackbone = Literal["medicalnet_resnet18", "monai_densenet121"]


@dataclass
class ModelConfig:
    backbone: ModelBackbone = "medicalnet_resnet18"
    num_classes: int = 2
    weights_dir: Path = Path("weights")


MEDICALNET_URL = "https://github.com/Tencent/MedicalNet/releases/download/v1.0/resnet_18_23dataset.pth"


def _load_medicalnet(model: torch.nn.Module, weights_dir: Path) -> torch.nn.Module:
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "resnet_18_23dataset.pth"

    if not checkpoint_path.exists():
        torch.hub.download_url_to_file(MEDICALNET_URL, str(checkpoint_path))

    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)

    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        if key.startswith("fc."):
            continue
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        print(f"Ignored unexpected MedicalNet keys: {unexpected}")
    if missing:
        print(f"Missing MedicalNet keys: {missing}")
    return model


def build_model(config: ModelConfig) -> torch.nn.Module:
    if config.backbone == "medicalnet_resnet18":
        model = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=config.num_classes,
        )
        model = _load_medicalnet(model, config.weights_dir)
        return model

    if config.backbone == "monai_densenet121":
        return DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=config.num_classes,
        )

    raise ValueError(f"Unsupported backbone: {config.backbone}")
