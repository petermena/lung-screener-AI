from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom


def _slice_position(ds: pydicom.Dataset) -> float:
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return float(getattr(ds, "InstanceNumber", 0))


def load_study_volume(study_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    dcm_paths = sorted(study_dir.glob("*.dcm"))
    if not dcm_paths:
        raise ValueError(f"No DICOM files found in {study_dir}")

    slices = [pydicom.dcmread(p) for p in dcm_paths]
    slices.sort(key=_slice_position)

    arrays = [s.pixel_array.astype(np.float32) for s in slices]
    volume = np.stack(arrays)

    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    hu_volume = volume * slope + intercept

    spacing = [float(x) for x in getattr(slices[0], "PixelSpacing", [1.0, 1.0])]
    z_spacing = abs(_slice_position(slices[1]) - _slice_position(slices[0])) if len(slices) > 1 else 1.0

    metadata = {
        "num_slices": len(slices),
        "shape": list(hu_volume.shape),
        "slice_thickness": z_spacing,
        "pixel_spacing": spacing,
        "patient_id": str(getattr(slices[0], "PatientID", "unknown")),
        "study_instance_uid": str(getattr(slices[0], "StudyInstanceUID", "unknown")),
        "series_instance_uid": str(getattr(slices[0], "SeriesInstanceUID", "unknown")),
        "study_date": str(getattr(slices[0], "StudyDate", "unknown")),
        "modality": str(getattr(slices[0], "Modality", "CT")),
        "intensity_min": float(hu_volume.min()),
        "intensity_max": float(hu_volume.max()),
    }
    return hu_volume, metadata


def normalize_for_display(slice_2d: np.ndarray, window_center: int = -600, window_width: int = 1500) -> np.ndarray:
    low = window_center - (window_width / 2)
    high = window_center + (window_width / 2)
    clipped = np.clip(slice_2d, low, high)
    norm = ((clipped - low) / (high - low) * 255).astype(np.uint8)
    return norm
