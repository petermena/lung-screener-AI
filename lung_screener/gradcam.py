"""3D Grad-CAM saliency maps for lung nodule explainability.

Generates visual explanations showing which voxel regions drove the
model's nodule prediction. Useful for radiologist trust and debugging.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import NoduleDenseNet3D, NoduleResNet3D, build_model

logger = logging.getLogger(__name__)


class GradCAM3D:
    """Grad-CAM for 3D CNNs.

    Hooks into the last convolutional layer before global average pooling
    to compute class-discriminative saliency maps.

    Usage:
        gradcam = GradCAM3D(model)
        heatmap, prediction = gradcam.generate(patch_tensor)
        # heatmap shape: (D, H, W) values in [0, 1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        """Initialize GradCAM3D.

        Args:
            model: Trained NoduleResNet3D or NoduleDenseNet3D.
            target_layer: The layer to hook into. If None, auto-detected
                from model architecture (last conv stage before GAP).
        """
        self.model = model
        self.model.eval()

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Auto-detect target layer
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self) -> torch.nn.Module:
        """Auto-detect the last conv layer before GAP."""
        if isinstance(self.model, NoduleResNet3D):
            return self.model.stage4
        elif isinstance(self.model, NoduleDenseNet3D):
            # Last dense block (before final_bn and GAP)
            return self.model.blocks[-1]
        else:
            raise ValueError(
                f"Cannot auto-detect target layer for {type(self.model).__name__}. "
                "Pass target_layer explicitly."
            )

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Generate a Grad-CAM heatmap for a single 3D patch.

        Args:
            input_tensor: (1, 1, D, H, W) or (1, D, H, W) tensor.
            target_class: Class index to explain. Defaults to the
                predicted class (argmax of logits).

        Returns:
            Tuple of:
                - heatmap: (D, H, W) numpy array in [0, 1]
                - prediction: dict with 'class', 'confidence', 'logits'
        """
        # Ensure correct shape
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        assert input_tensor.dim() == 5, f"Expected 5D tensor, got {input_tensor.dim()}D"

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        logits = output["logits"]
        probs = F.softmax(logits, dim=1)

        pred_class = int(torch.argmax(logits, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Compute Grad-CAM weights: global average of gradients per channel
        gradients = self._gradients[0]  # (C, d, h, w)
        weights = gradients.mean(dim=(1, 2, 3))  # (C,)

        # Weighted combination of activation maps
        activations = self._activations[0]  # (C, d, h, w)
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — only keep positive influence
        cam = F.relu(cam)

        # Upsample to input spatial size
        spatial_size = input_tensor.shape[2:]  # (D, H, W)
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, d, h, w)
        cam = F.interpolate(cam, size=spatial_size, mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        prediction = {
            "class": pred_class,
            "confidence": confidence,
            "logits": logits[0].detach().cpu().numpy(),
            "target_class": target_class,
            "nodule_probability": float(probs[0, 1].item()),
        }

        return cam, prediction

    def release(self):
        """Remove hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def render_slices(
    patch: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    num_slices: int = 9,
    alpha: float = 0.4,
    prediction: dict | None = None,
):
    """Render 2D slice overlays of GradCAM heatmap on the CT patch.

    Produces a grid of axial slices with the heatmap overlaid in color.

    Args:
        patch: (D, H, W) numpy array — the CT patch (normalized [0,1]).
        heatmap: (D, H, W) numpy array — GradCAM output in [0,1].
        output_path: Where to save the PNG.
        num_slices: Number of evenly-spaced axial slices to show.
        alpha: Heatmap overlay transparency (0=invisible, 1=opaque).
        prediction: Optional prediction dict to show in title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = patch.shape[0]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    cols = min(num_slices, 3)
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_slices == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_slices:
            idx = indices[i]
            ct_slice = patch[idx]
            cam_slice = heatmap[idx]

            ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam_slice, cmap="jet", alpha=alpha, vmin=0, vmax=1)
            ax.set_title(f"Slice {idx}", fontsize=10)
        ax.axis("off")

    # Title
    if prediction:
        label = "NODULE" if prediction["class"] == 1 else "NON-NODULE"
        conf = prediction["confidence"]
        fig.suptitle(
            f"GradCAM — {label} (conf: {conf:.1%})",
            fontsize=14,
            fontweight="bold",
        )
    else:
        fig.suptitle("GradCAM Saliency Map", fontsize=14)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"GradCAM visualization saved to {output_path}")
    return output_path


def render_three_plane(
    patch: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
    prediction: dict | None = None,
):
    """Render axial, coronal, and sagittal views through the center.

    Produces a 1x3 figure showing the center slice in each plane,
    with the GradCAM heatmap overlaid.

    Args:
        patch: (D, H, W) numpy array.
        heatmap: (D, H, W) numpy array.
        output_path: Where to save the PNG.
        alpha: Heatmap overlay transparency.
        prediction: Optional prediction dict.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, h, w = patch.shape

    slices = [
        ("Axial", patch[d // 2], heatmap[d // 2]),
        ("Coronal", patch[:, h // 2, :], heatmap[:, h // 2, :]),
        ("Sagittal", patch[:, :, w // 2], heatmap[:, :, w // 2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, ct_sl, cam_sl) in zip(axes, slices):
        ax.imshow(ct_sl, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam_sl, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    if prediction:
        label = "NODULE" if prediction["class"] == 1 else "NON-NODULE"
        conf = prediction["confidence"]
        fig.suptitle(
            f"GradCAM — {label} (conf: {conf:.1%})",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Three-plane GradCAM saved to {output_path}")
    return output_path


def load_model_for_gradcam(
    config: dict,
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Load a trained model in eval mode for GradCAM analysis.

    Args:
        config: Model configuration dict.
        checkpoint_path: Path to .pth checkpoint.
        device: Device string. Auto-detected if None.

    Returns:
        Tuple of (model, device).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(dev)

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} on {dev}")
    return model, dev
