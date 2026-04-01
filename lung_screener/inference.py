from __future__ import annotations

from pathlib import Path

import torch

from lung_screener.data import load_volume, preprocess_volume
from lung_screener.model import ModelConfig, build_model


@torch.inference_mode()
def predict_scan(
    scan_path: Path,
    checkpoint_path: Path,
    backbone: str = "medicalnet_resnet18",
    device: str = "cuda",
) -> dict[str, float]:
    model = build_model(ModelConfig(backbone=backbone))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    model = model.to(device)
    model.eval()

    volume = load_volume(scan_path)
    x = preprocess_volume(volume).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    return {
        "negative": float(probs[0]),
        "suspicious": float(probs[1]),
    }
