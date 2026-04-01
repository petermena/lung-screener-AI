from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F


SUPPORTED_EXTENSIONS = {".nii", ".nii.gz", ".mha", ".mhd"}


def _is_nifti(path: Path) -> bool:
    return any(str(path).endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def load_volume(path: Path) -> np.ndarray:
    """Load a CT volume from a NIfTI/MHA file or a directory of DICOM slices."""
    path = Path(path)

    if path.is_file() and _is_nifti(path):
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image).astype(np.float32)

    if path.is_dir():
        dcm_files = sorted(path.glob("*.dcm"))
        if not dcm_files:
            raise ValueError(f"No DICOM slices found under: {path}")

        slices = [pydicom.dcmread(str(file), force=True) for file in dcm_files]
        slices = sorted(slices, key=lambda s: float(getattr(s, "ImagePositionPatient", [0, 0, 0])[2]))

        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

        slope = float(getattr(slices[0], "RescaleSlope", 1.0))
        intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
        volume = volume * slope + intercept
        return volume

    raise ValueError("Input path must be a .nii/.mha file or a directory containing .dcm files.")


def preprocess_volume(
    volume: np.ndarray,
    out_shape: tuple[int, int, int] = (96, 192, 192),
    hu_window: tuple[int, int] = (-1200, 600),
) -> torch.Tensor:
    """Normalize and resize to [1, D, H, W]."""
    low, high = hu_window
    volume = np.clip(volume, low, high)
    volume = (volume - low) / float(high - low)

    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=out_shape, mode="trilinear", align_corners=False)
    return tensor.squeeze(0)


def middle_slices(volume: np.ndarray, count: int = 9) -> Iterable[np.ndarray]:
    d = volume.shape[0]
    center = d // 2
    half = count // 2
    indices = [min(max(center + i, 0), d - 1) for i in range(-half, half + 1)]
    return [volume[i] for i in indices]
