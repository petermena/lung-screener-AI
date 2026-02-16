"""DICOM ingestion and CT scan preprocessing pipeline.

Handles loading DICOM series, resampling to isotropic spacing,
Hounsfield Unit windowing, and lung segmentation for candidate extraction.
"""

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology


def load_dicom_series(dicom_dir: str | Path) -> sitk.Image:
    """Load a DICOM series from a directory into a SimpleITK image.

    Args:
        dicom_dir: Path to directory containing DICOM files for one series.

    Returns:
        SimpleITK image with correct spatial metadata.

    Raises:
        FileNotFoundError: If no DICOM files found in directory.
        RuntimeError: If DICOM series cannot be read.
    """
    dicom_dir = Path(dicom_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    # Use the first series (typically only one per directory)
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    return image


def load_mhd(mhd_path: str | Path) -> sitk.Image:
    """Load a .mhd/.raw volume (LUNA16 format).

    Args:
        mhd_path: Path to .mhd header file.

    Returns:
        SimpleITK image.
    """
    return sitk.ReadImage(str(mhd_path))


def resample_volume(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    """Resample a volume to isotropic target spacing.

    Args:
        image: Input SimpleITK image.
        target_spacing: Desired voxel spacing in mm (z, y, x).

    Returns:
        Resampled SimpleITK image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new size to maintain field of view
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)


def apply_hu_window(
    volume: np.ndarray,
    hu_min: float = -1200.0,
    hu_max: float = 600.0,
    normalize: bool = True,
) -> np.ndarray:
    """Apply Hounsfield Unit windowing and optional normalization.

    Args:
        volume: 3D numpy array of HU values.
        hu_min: Lower HU bound (below is clipped).
        hu_max: Upper HU bound (above is clipped).
        normalize: If True, scale to [0, 1].

    Returns:
        Windowed (and optionally normalized) volume.
    """
    volume = np.clip(volume, hu_min, hu_max)
    if normalize:
        volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32)


def segment_lungs(volume: np.ndarray, hu_threshold: float = -400.0) -> np.ndarray:
    """Segment lung tissue from a CT volume using thresholding and morphology.

    Args:
        volume: 3D numpy array in Hounsfield Units (before normalization).
        hu_threshold: HU threshold to separate air/lung from tissue.

    Returns:
        Binary mask (1 = lung region, 0 = background).
    """
    # Threshold to get air regions
    binary = volume < hu_threshold

    # Clear border-connected components (exterior air)
    cleared = np.copy(binary)
    for ax in range(3):
        for sl in [0, -1]:
            slices = [slice(None)] * 3
            slices[ax] = sl
            seed = np.zeros_like(binary)
            seed[tuple(slices)] = binary[tuple(slices)]
            # Flood fill from this face
            filled = ndimage.binary_dilation(seed, iterations=0, mask=binary)
            cleared[filled] = False

    # Label connected components and keep the two largest (left + right lung)
    labels = measure.label(cleared)
    regions = measure.regionprops(labels)

    if len(regions) < 1:
        # Fallback: return the thresholded mask
        return binary.astype(np.float32)

    # Sort by area, keep top 2
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    lung_mask = np.zeros_like(cleared, dtype=bool)
    for region in regions[:2]:
        lung_mask[labels == region.label] = True

    # Morphological closing to fill small holes
    struct = morphology.ball(5)
    # Process slice-by-slice if 3D closing is too expensive
    lung_mask = ndimage.binary_closing(lung_mask, structure=struct, iterations=1)
    lung_mask = ndimage.binary_fill_holes(lung_mask)

    return lung_mask.astype(np.float32)


def extract_candidates(
    volume: np.ndarray,
    lung_mask: np.ndarray,
    spacing: tuple[float, float, float],
    min_size_mm: float = 3.0,
    max_size_mm: float = 30.0,
) -> list[dict]:
    """Extract nodule candidate locations from a lung-masked volume.

    Uses connected component analysis on thresholded regions within the lung
    mask to find potential nodule candidates.

    Args:
        volume: Normalized 3D volume [0, 1].
        lung_mask: Binary lung segmentation mask.
        spacing: Voxel spacing in mm (z, y, x).
        min_size_mm: Minimum candidate diameter in mm.
        max_size_mm: Maximum candidate diameter in mm.

    Returns:
        List of candidate dicts with keys:
            - center_voxel: (z, y, x) voxel coordinates
            - center_world: (z, y, x) world coordinates in mm
            - diameter_mm: estimated diameter
    """
    spacing = np.array(spacing)

    # Look for bright regions within lung mask (potential nodules are denser than lung parenchyma)
    masked = volume * lung_mask
    threshold = 0.4  # Corresponds to roughly -480 HU in default window
    candidates_mask = masked > threshold

    # Remove very small noise
    candidates_mask = morphology.remove_small_objects(candidates_mask, max_size=10)

    labels = measure.label(candidates_mask)
    regions = measure.regionprops(labels)

    candidates = []
    for region in regions:
        # Estimate diameter from volume
        volume_mm3 = region.area * np.prod(spacing)
        diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

        if min_size_mm <= diameter_mm <= max_size_mm:
            centroid = np.array(region.centroid)
            center_world = centroid * spacing

            candidates.append({
                "center_voxel": tuple(centroid.astype(int)),
                "center_world": tuple(center_world),
                "diameter_mm": float(diameter_mm),
            })

    return candidates


def extract_patch(
    volume: np.ndarray,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int] = (48, 48, 48),
) -> np.ndarray:
    """Extract a 3D patch centered at the given voxel coordinates.

    Handles boundary conditions by zero-padding.

    Args:
        volume: 3D numpy array.
        center: (z, y, x) center voxel.
        patch_size: (d, h, w) patch dimensions.

    Returns:
        3D numpy array of shape patch_size.
    """
    d, h, w = patch_size
    cz, cy, cx = center
    vol_shape = volume.shape

    # Compute source and destination slices
    patch = np.zeros(patch_size, dtype=volume.dtype)

    # Source ranges (clipped to volume bounds)
    sz_start = max(0, cz - d // 2)
    sz_end = min(vol_shape[0], cz + d // 2)
    sy_start = max(0, cy - h // 2)
    sy_end = min(vol_shape[1], cy + h // 2)
    sx_start = max(0, cx - w // 2)
    sx_end = min(vol_shape[2], cx + w // 2)

    # Destination ranges
    dz_start = sz_start - (cz - d // 2)
    dz_end = dz_start + (sz_end - sz_start)
    dy_start = sy_start - (cy - h // 2)
    dy_end = dy_start + (sy_end - sy_start)
    dx_start = sx_start - (cx - w // 2)
    dx_end = dx_start + (sx_end - sx_start)

    patch[dz_start:dz_end, dy_start:dy_end, dx_start:dx_end] = volume[
        sz_start:sz_end, sy_start:sy_end, sx_start:sx_end
    ]

    return patch


class CTPreprocessor:
    """End-to-end CT scan preprocessor.

    Orchestrates loading, resampling, windowing, lung segmentation,
    and candidate extraction.
    """

    def __init__(self, config: dict):
        self.target_spacing = tuple(config.get("preprocessing", {}).get(
            "target_spacing", [1.0, 1.0, 1.0]
        ))
        hu_window = config.get("preprocessing", {}).get("hu_window", {})
        self.hu_min = hu_window.get("min", -1200)
        self.hu_max = hu_window.get("max", 600)
        self.normalize = config.get("preprocessing", {}).get("normalize", True)
        self.patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))

        cand_config = config.get("candidate_detection", {})
        self.lung_threshold = cand_config.get("lung_threshold", -400)
        self.min_size_mm = cand_config.get("min_size_mm", 3.0)
        self.max_size_mm = cand_config.get("max_size_mm", 30.0)

    def process_scan(self, image: sitk.Image) -> dict:
        """Process a single CT scan through the full pipeline.

        Args:
            image: SimpleITK image of the CT scan.

        Returns:
            Dict with keys:
                - volume: preprocessed normalized volume
                - volume_hu: raw Hounsfield Unit volume (for calcification analysis)
                - lung_mask: binary lung segmentation
                - candidates: list of candidate dicts
                - spacing: final voxel spacing
                - origin: world origin
        """
        # Resample to isotropic spacing
        resampled = resample_volume(image, self.target_spacing)
        spacing = resampled.GetSpacing()
        origin = resampled.GetOrigin()

        # Convert to numpy (HU values)
        volume_hu = sitk.GetArrayFromImage(resampled).astype(np.float32)

        # Segment lungs before HU windowing
        lung_mask = segment_lungs(volume_hu, self.lung_threshold)

        # Apply HU window and normalize
        volume = apply_hu_window(volume_hu, self.hu_min, self.hu_max, self.normalize)

        # Extract candidates
        candidates = extract_candidates(
            volume, lung_mask, spacing,
            self.min_size_mm, self.max_size_mm,
        )

        return {
            "volume": volume,
            "volume_hu": volume_hu,
            "lung_mask": lung_mask,
            "candidates": candidates,
            "spacing": spacing,
            "origin": origin,
        }

    def extract_candidate_patches(
        self, volume: np.ndarray, candidates: list[dict]
    ) -> np.ndarray:
        """Extract 3D patches for all candidates.

        Args:
            volume: Preprocessed 3D volume.
            candidates: List of candidate dicts from extract_candidates.

        Returns:
            Array of shape (N, 1, D, H, W) ready for model input.
        """
        patches = []
        for cand in candidates:
            patch = extract_patch(volume, cand["center_voxel"], self.patch_size)
            patches.append(patch)

        if not patches:
            return np.empty((0, 1, *self.patch_size), dtype=np.float32)

        # Stack and add channel dimension
        patches = np.stack(patches, axis=0)[:, np.newaxis, ...]
        return patches
