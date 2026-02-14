"""Export trained model to ONNX format for lightweight offline deployment.

ONNX models can run via onnxruntime without needing the full PyTorch stack,
significantly reducing the install size (~50MB vs ~2GB).
"""

import logging
from pathlib import Path

import torch

from .model import build_model

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: dict,
    opset_version: int = 17,
) -> Path:
    """Export a trained checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        output_path: Where to save the .onnx file.
        config: Model configuration dict.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)

    # Load model
    device = torch.device("cpu")
    model = build_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create dummy input matching expected patch size
    patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))
    dummy_input = torch.randn(1, 1, *patch_size)

    # Export
    logger.info(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )

    logger.info(f"ONNX model saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_path
