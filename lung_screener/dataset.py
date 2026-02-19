"""Dataset classes for LUNA16/LUNA25/LIDC-IDRI lung nodule data.

Handles loading annotations, creating train/val splits, and providing
3D patches with enhanced data augmentation for training.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import ConcatDataset, Dataset

from .preprocessing import CTPreprocessor, extract_patch, load_mhd

logger = logging.getLogger(__name__)


class LUNA16Dataset(Dataset):
    """Dataset for LUNA16 challenge data.

    Expects the LUNA16 directory structure:
        dataset_dir/
            subset0/ ... subset9/   (CT volumes as .mhd/.raw)
            annotations.csv          (nodule annotations)
            candidates_V2.csv        (candidate locations)

    Each sample is a 3D patch centered on a candidate location,
    labeled as nodule (1) or non-nodule (0).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        config: dict,
        split: str = "train",
        val_split: float = 0.2,
        augment: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.split = split
        self.augment = augment and split == "train"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preprocessor = CTPreprocessor(config)
        self.patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))

        aug_config = config.get("training", {}).get("augmentation", {})
        self.aug_rotation = aug_config.get("rotation", True)
        self.aug_flip = aug_config.get("flip", True)
        self.aug_scale_range = aug_config.get("scale_range", [0.9, 1.1])
        self.aug_noise_std = aug_config.get("noise_std", 0.01)
        # Enhanced augmentation options
        self.aug_elastic = aug_config.get("elastic_deformation", False)
        self.aug_elastic_alpha = aug_config.get("elastic_alpha", 15.0)
        self.aug_elastic_sigma = aug_config.get("elastic_sigma", 3.0)
        self.aug_intensity_shift = aug_config.get("intensity_shift", 0.0)
        self.aug_intensity_scale = aug_config.get("intensity_scale", 0.0)
        self.aug_mixup_alpha = aug_config.get("mixup_alpha", 0.0)

        # Load annotations and candidates
        self.samples = self._load_samples(val_split)

        # Cache for loaded volumes (seriesuid -> volume)
        self._volume_cache: dict[str, np.ndarray] = {}

    def _load_samples(self, val_split: float) -> list[dict]:
        """Load and merge annotations with candidates, then split."""
        annotations_path = self.dataset_dir / "annotations.csv"
        candidates_path = self.dataset_dir / "candidates_V2.csv"

        samples = []

        if annotations_path.exists() and candidates_path.exists():
            annotations = pd.read_csv(annotations_path)
            candidates = pd.read_csv(candidates_path)

            for _, row in candidates.iterrows():
                sample = {
                    "seriesuid": row["seriesuid"],
                    "coord_x": float(row["coordX"]),
                    "coord_y": float(row["coordY"]),
                    "coord_z": float(row["coordZ"]),
                    "label": int(row["class"]),
                }

                # If positive, find the matching annotation for diameter
                if sample["label"] == 1:
                    matching = annotations[
                        (annotations["seriesuid"] == row["seriesuid"])
                        & (
                            (annotations["coordX"] - row["coordX"]).abs() < 5.0
                        )
                        & (
                            (annotations["coordY"] - row["coordY"]).abs() < 5.0
                        )
                        & (
                            (annotations["coordZ"] - row["coordZ"]).abs() < 5.0
                        )
                    ]
                    if len(matching) > 0:
                        sample["diameter_mm"] = float(matching.iloc[0]["diameter_mm"])
                    else:
                        sample["diameter_mm"] = 0.0
                else:
                    sample["diameter_mm"] = 0.0

                samples.append(sample)
        else:
            # If no annotation files, return empty dataset
            return []

        # Deterministic split based on seriesuid hash
        unique_series = sorted(set(s["seriesuid"] for s in samples))
        np.random.seed(42)
        np.random.shuffle(unique_series)
        split_idx = int(len(unique_series) * (1 - val_split))

        if self.split == "train":
            valid_series = set(unique_series[:split_idx])
        else:
            valid_series = set(unique_series[split_idx:])

        samples = [s for s in samples if s["seriesuid"] in valid_series]

        # Balance positive/negative samples for training
        if self.split == "train":
            samples = self._balance_samples(samples)
        else:
            # Cap validation candidates for faster evaluation
            max_val = self.config.get("data", {}).get("max_val_candidates", 0)
            if max_val > 0 and len(samples) > max_val:
                samples = self._subsample_val(samples, max_val)

        return samples

    def _balance_samples(self, samples: list[dict]) -> list[dict]:
        """Balance positive and negative samples based on config ratio."""
        pos_neg_ratio = self.config.get("training", {}).get("pos_neg_ratio", 1.0)
        positives = [s for s in samples if s["label"] == 1]
        negatives = [s for s in samples if s["label"] == 0]

        if not positives:
            return samples

        # Undersample negatives to match desired ratio
        max_negatives = int(len(positives) / pos_neg_ratio)
        if len(negatives) > max_negatives:
            np.random.seed(42)
            indices = np.random.choice(len(negatives), max_negatives, replace=False)
            negatives = [negatives[i] for i in indices]

        return positives + negatives

    def _subsample_val(self, samples: list[dict], max_candidates: int) -> list[dict]:
        """Subsample validation set while keeping all positives."""
        positives = [s for s in samples if s["label"] == 1]
        negatives = [s for s in samples if s["label"] == 0]

        max_neg = max(max_candidates - len(positives), 0)
        if len(negatives) > max_neg:
            np.random.seed(42)
            indices = np.random.choice(len(negatives), max_neg, replace=False)
            negatives = [negatives[i] for i in indices]

        return positives + negatives

    def _find_volume_path(self, seriesuid: str) -> Path | None:
        """Find the .mhd file for a given series UID across subsets."""
        for subset_dir in sorted(self.dataset_dir.glob("subset*")):
            mhd_path = subset_dir / f"{seriesuid}.mhd"
            if mhd_path.exists():
                return mhd_path
        return None

    def _load_volume(self, seriesuid: str) -> np.ndarray | None:
        """Load and preprocess a volume, with caching."""
        if seriesuid in self._volume_cache:
            return self._volume_cache[seriesuid]

        # Check disk cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{seriesuid}.npy"
            if cache_path.exists():
                volume = np.load(cache_path).astype(np.float32)
                self._volume_cache[seriesuid] = volume
                return volume

        # Load from raw data
        mhd_path = self._find_volume_path(seriesuid)
        if mhd_path is None:
            return None

        image = load_mhd(mhd_path)
        result = self.preprocessor.process_scan(image, training_mode=True)
        volume = result["volume"]

        # Save to disk cache (atomic write to avoid corruption from parallel workers)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_dir / f"{seriesuid}.npy.tmp.{os.getpid()}"
            cache_path = self.cache_dir / f"{seriesuid}.npy"
            np.save(tmp_path, volume.astype(np.float16))
            try:
                os.replace(tmp_path, cache_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)

        # Keep in memory cache (limit to ~50 volumes; ~300MB each ≈ 15GB max)
        if len(self._volume_cache) < 50:
            self._volume_cache[seriesuid] = volume

        return volume

    def _world_to_voxel(
        self, world_coord: tuple[float, float, float], origin: tuple, spacing: tuple
    ) -> tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        voxel = tuple(
            int(round((w - o) / s)) for w, o, s in zip(world_coord, origin, spacing)
        )
        return voxel

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        """Apply random data augmentation to a 3D patch.

        Supports standard augmentations (rotation, flip, scale, noise)
        plus enhanced options (elastic deformation, intensity shifts).
        """
        # Random rotation (90-degree increments around each axis)
        if self.aug_rotation:
            k = np.random.randint(0, 4)
            axes = [(0, 1), (0, 2), (1, 2)]
            ax = axes[np.random.randint(0, 3)]
            patch = np.rot90(patch, k=k, axes=ax).copy()

        # Random flips
        if self.aug_flip:
            for axis in range(3):
                if np.random.random() > 0.5:
                    patch = np.flip(patch, axis=axis).copy()

        # Random scaling
        if self.aug_scale_range:
            scale = np.random.uniform(*self.aug_scale_range)
            if abs(scale - 1.0) > 0.01:
                patch = ndimage.zoom(patch, scale, order=1)
                # Crop or pad back to original size
                patch = self._crop_or_pad(patch, self.patch_size)

        # Elastic deformation
        if self.aug_elastic:
            patch = self._elastic_deformation(
                patch, self.aug_elastic_alpha, self.aug_elastic_sigma
            )

        # Random intensity shift
        if self.aug_intensity_shift > 0:
            shift = np.random.uniform(
                -self.aug_intensity_shift, self.aug_intensity_shift
            )
            patch = np.clip(patch + shift, 0.0, 1.0)

        # Random intensity scale
        if self.aug_intensity_scale > 0:
            scale = np.random.uniform(
                1.0 - self.aug_intensity_scale, 1.0 + self.aug_intensity_scale
            )
            patch = np.clip(patch * scale, 0.0, 1.0)

        # Random Gaussian noise
        if self.aug_noise_std > 0:
            noise = np.random.normal(0, self.aug_noise_std, patch.shape).astype(np.float32)
            patch = patch + noise

        return patch

    def _elastic_deformation(
        self, patch: np.ndarray, alpha: float, sigma: float
    ) -> np.ndarray:
        """Apply random elastic deformation to a 3D patch.

        Creates smooth random displacement fields and applies them to
        the patch for realistic tissue-like deformations.

        Args:
            patch: 3D numpy array.
            alpha: Deformation magnitude.
            sigma: Gaussian smoothing sigma for displacement fields.

        Returns:
            Deformed patch.
        """
        shape = patch.shape
        # Generate random displacement fields
        dz = ndimage.gaussian_filter(
            np.random.randn(*shape) * alpha, sigma, mode="reflect"
        )
        dy = ndimage.gaussian_filter(
            np.random.randn(*shape) * alpha, sigma, mode="reflect"
        )
        dx = ndimage.gaussian_filter(
            np.random.randn(*shape) * alpha, sigma, mode="reflect"
        )

        # Create coordinate grids
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )

        # Apply displacements
        coords = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1),
        ]

        return ndimage.map_coordinates(
            patch, coords, order=1, mode="reflect"
        ).astype(patch.dtype)

    def _crop_or_pad(self, volume: np.ndarray, target_size: tuple) -> np.ndarray:
        """Crop or zero-pad a volume to the target size."""
        result = np.zeros(target_size, dtype=volume.dtype)
        # Compute overlap
        slices_src = []
        slices_dst = []
        for i in range(3):
            if volume.shape[i] > target_size[i]:
                start = (volume.shape[i] - target_size[i]) // 2
                slices_src.append(slice(start, start + target_size[i]))
                slices_dst.append(slice(0, target_size[i]))
            else:
                start = (target_size[i] - volume.shape[i]) // 2
                slices_src.append(slice(0, volume.shape[i]))
                slices_dst.append(slice(start, start + volume.shape[i]))

        result[tuple(slices_dst)] = volume[tuple(slices_src)]
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load volume
        volume = self._load_volume(sample["seriesuid"])
        if volume is None:
            # Return zeros if volume can't be loaded (shouldn't happen in practice)
            patch = np.zeros(self.patch_size, dtype=np.float32)
        else:
            # The coordinates in LUNA16 are world coordinates (x, y, z)
            # SimpleITK uses (x, y, z) ordering, numpy uses (z, y, x)
            center_voxel = (
                int(round(sample["coord_z"])),
                int(round(sample["coord_y"])),
                int(round(sample["coord_x"])),
            )
            # Clamp to volume bounds
            center_voxel = tuple(
                max(0, min(c, s - 1)) for c, s in zip(center_voxel, volume.shape)
            )
            patch = extract_patch(volume, center_voxel, self.patch_size)

        # Augmentation
        if self.augment:
            patch = self._augment_patch(patch)

        # Convert to tensor: (1, D, H, W)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "patch": patch_tensor,
            "label": label_tensor,
            "seriesuid": sample["seriesuid"],
            "diameter_mm": sample["diameter_mm"],
        }


class LUNA25Dataset(Dataset):
    """Dataset for LUNA25 challenge data.

    Supports two directory layouts:

    1. **Nodule blocks** (official LUNA25 baseline format):
           dataset_dir/
               annotations.csv          (AnnotationID, label columns)
               image/{AnnotationID}.npy  (pre-extracted 3D patches)
               metadata/{AnnotationID}.npy (origin, spacing, transform)

    2. **Full CT volumes** (raw MHA/MHD scans + annotation CSV):
           dataset_dir/
               images/                  (CT volumes as .mha or .mhd)
               annotations.csv          (seriesuid, coordX/Y/Z, diameter_mm, label)

    The format is auto-detected based on the presence of an ``image/``
    subdirectory.  Each sample is a 3D patch labeled as malignant (1) or
    benign (0).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        config: dict,
        split: str = "train",
        val_split: float = 0.2,
        augment: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.split = split
        self.augment = augment and split == "train"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preprocessor = CTPreprocessor(config)
        self.patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))

        aug_config = config.get("training", {}).get("augmentation", {})
        self.aug_rotation = aug_config.get("rotation", True)
        self.aug_flip = aug_config.get("flip", True)
        self.aug_scale_range = aug_config.get("scale_range", [0.9, 1.1])
        self.aug_noise_std = aug_config.get("noise_std", 0.01)
        self.aug_elastic = aug_config.get("elastic_deformation", False)
        self.aug_elastic_alpha = aug_config.get("elastic_alpha", 15.0)
        self.aug_elastic_sigma = aug_config.get("elastic_sigma", 3.0)
        self.aug_intensity_shift = aug_config.get("intensity_shift", 0.0)
        self.aug_intensity_scale = aug_config.get("intensity_scale", 0.0)

        # Detect layout: nodule blocks vs full volumes
        self.nodule_blocks = (self.dataset_dir / "image").is_dir()

        self.samples = self._load_samples(val_split)

        # Volume cache (only used for full-volume mode)
        self._volume_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_samples(self, val_split: float) -> list[dict]:
        """Load annotation CSV and create train/val split."""
        annotations_path = self.dataset_dir / "annotations.csv"
        if not annotations_path.exists():
            logger.warning("LUNA25: annotations.csv not found in %s", self.dataset_dir)
            return []

        df = pd.read_csv(annotations_path)

        if self.nodule_blocks:
            samples = self._samples_from_blocks(df)
        else:
            samples = self._samples_from_volumes(df)

        if not samples:
            return []

        # Deterministic train/val split keyed on a unique ID per sample
        if self.nodule_blocks:
            unique_keys = sorted(set(s["annotation_id"] for s in samples))
        else:
            unique_keys = sorted(set(s["seriesuid"] for s in samples))

        np.random.seed(42)
        np.random.shuffle(unique_keys)
        split_idx = int(len(unique_keys) * (1 - val_split))

        if self.split == "train":
            valid_keys = set(unique_keys[:split_idx])
        else:
            valid_keys = set(unique_keys[split_idx:])

        key_field = "annotation_id" if self.nodule_blocks else "seriesuid"
        samples = [s for s in samples if s[key_field] in valid_keys]

        # Balance for training
        if self.split == "train":
            samples = self._balance_samples(samples)
        else:
            max_val = self.config.get("data", {}).get("max_val_candidates", 0)
            if max_val > 0 and len(samples) > max_val:
                samples = self._subsample_val(samples, max_val)

        return samples

    @staticmethod
    def _col(df: pd.DataFrame, *candidates: str) -> str:
        """Return the first column name that exists in *df* (case-insensitive)."""
        lower_map = {c.lower(): c for c in df.columns}
        for name in candidates:
            if name in df.columns:
                return name
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")

    def _samples_from_blocks(self, df: pd.DataFrame) -> list[dict]:
        """Build samples list from pre-extracted nodule blocks."""
        id_col = self._col(df, "AnnotationID", "annotation_id")
        label_col = self._col(df, "label", "class")
        series_col = self._col(df, "seriesuid", "SeriesInstanceUID") if any(
            c.lower() in ("seriesuid", "seriesinstanceuid") for c in df.columns
        ) else None
        has_diameter = any(c.lower() == "diameter_mm" for c in df.columns)
        diam_col = self._col(df, "diameter_mm") if has_diameter else None

        samples = []
        for _, row in df.iterrows():
            ann_id = str(row[id_col])
            npy_path = self.dataset_dir / "image" / f"{ann_id}.npy"
            if not npy_path.exists():
                continue
            samples.append({
                "annotation_id": ann_id,
                "seriesuid": str(row[series_col]) if series_col else ann_id,
                "label": int(row[label_col]),
                "diameter_mm": float(row[diam_col]) if diam_col else 0.0,
            })
        return samples

    def _samples_from_volumes(self, df: pd.DataFrame) -> list[dict]:
        """Build samples list from full-volume annotation CSV."""
        label_col = self._col(df, "label", "class")
        series_col = self._col(df, "seriesuid", "SeriesInstanceUID")
        coord_x_col = self._col(df, "coordX", "CoordX")
        coord_y_col = self._col(df, "coordY", "CoordY")
        coord_z_col = self._col(df, "coordZ", "CoordZ")
        id_col = self._col(df, "AnnotationID", "annotation_id") if any(
            c.lower() in ("annotationid", "annotation_id") for c in df.columns
        ) else None
        has_diameter = any(c.lower() == "diameter_mm" for c in df.columns)
        diam_col = self._col(df, "diameter_mm") if has_diameter else None

        samples = []
        for _, row in df.iterrows():
            samples.append({
                "annotation_id": str(row[id_col]) if id_col else "",
                "seriesuid": str(row[series_col]),
                "coord_x": float(row[coord_x_col]),
                "coord_y": float(row[coord_y_col]),
                "coord_z": float(row[coord_z_col]),
                "label": int(row[label_col]),
                "diameter_mm": float(row[diam_col]) if diam_col else 0.0,
            })
        return samples

    # ------------------------------------------------------------------
    # Balancing helpers (same logic as LUNA16Dataset)
    # ------------------------------------------------------------------

    def _balance_samples(self, samples: list[dict]) -> list[dict]:
        pos_neg_ratio = self.config.get("training", {}).get("pos_neg_ratio", 1.0)
        positives = [s for s in samples if s["label"] == 1]
        negatives = [s for s in samples if s["label"] == 0]
        if not positives:
            return samples
        max_negatives = int(len(positives) / pos_neg_ratio)
        if len(negatives) > max_negatives:
            np.random.seed(42)
            indices = np.random.choice(len(negatives), max_negatives, replace=False)
            negatives = [negatives[i] for i in indices]
        return positives + negatives

    def _subsample_val(self, samples: list[dict], max_candidates: int) -> list[dict]:
        positives = [s for s in samples if s["label"] == 1]
        negatives = [s for s in samples if s["label"] == 0]
        max_neg = max(max_candidates - len(positives), 0)
        if len(negatives) > max_neg:
            np.random.seed(42)
            indices = np.random.choice(len(negatives), max_neg, replace=False)
            negatives = [negatives[i] for i in indices]
        return positives + negatives

    # ------------------------------------------------------------------
    # Volume / patch loading
    # ------------------------------------------------------------------

    def _load_block(self, annotation_id: str) -> np.ndarray | None:
        """Load a pre-extracted nodule block (.npy)."""
        npy_path = self.dataset_dir / "image" / f"{annotation_id}.npy"
        if not npy_path.exists():
            return None
        return np.load(npy_path, mmap_mode="r").astype(np.float32)

    def _find_volume_path(self, seriesuid: str) -> Path | None:
        """Find a .mha or .mhd file for *seriesuid*."""
        images_dir = self.dataset_dir / "images"
        if images_dir.is_dir():
            for ext in (".mha", ".mhd"):
                p = images_dir / f"{seriesuid}{ext}"
                if p.exists():
                    return p
        # Fallback: scan subset* dirs like LUNA16
        for subset_dir in sorted(self.dataset_dir.glob("subset*")):
            for ext in (".mha", ".mhd"):
                p = subset_dir / f"{seriesuid}{ext}"
                if p.exists():
                    return p
        return None

    def _load_volume(self, seriesuid: str) -> np.ndarray | None:
        if seriesuid in self._volume_cache:
            return self._volume_cache[seriesuid]

        if self.cache_dir:
            cache_path = self.cache_dir / f"luna25_{seriesuid}.npy"
            if cache_path.exists():
                volume = np.load(cache_path).astype(np.float32)
                self._volume_cache[seriesuid] = volume
                return volume

        vol_path = self._find_volume_path(seriesuid)
        if vol_path is None:
            return None

        # SimpleITK reads both .mha and .mhd transparently
        image = load_mhd(vol_path)
        result = self.preprocessor.process_scan(image, training_mode=True)
        volume = result["volume"]

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_dir / f"luna25_{seriesuid}.npy.tmp.{os.getpid()}"
            cache_path = self.cache_dir / f"luna25_{seriesuid}.npy"
            np.save(tmp_path, volume.astype(np.float16))
            try:
                os.replace(tmp_path, cache_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)

        if len(self._volume_cache) < 50:
            self._volume_cache[seriesuid] = volume

        return volume

    # ------------------------------------------------------------------
    # Augmentation (delegates to LUNA16Dataset helpers)
    # ------------------------------------------------------------------

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        """Reuse the same augmentation pipeline as LUNA16Dataset."""
        if self.aug_rotation:
            k = np.random.randint(0, 4)
            axes = [(0, 1), (0, 2), (1, 2)]
            ax = axes[np.random.randint(0, 3)]
            patch = np.rot90(patch, k=k, axes=ax).copy()

        if self.aug_flip:
            for axis in range(3):
                if np.random.random() > 0.5:
                    patch = np.flip(patch, axis=axis).copy()

        if self.aug_scale_range:
            scale = np.random.uniform(*self.aug_scale_range)
            if abs(scale - 1.0) > 0.01:
                patch = ndimage.zoom(patch, scale, order=1)
                patch = _crop_or_pad(patch, self.patch_size)

        if self.aug_elastic:
            patch = _elastic_deformation(
                patch, self.aug_elastic_alpha, self.aug_elastic_sigma
            )

        if self.aug_intensity_shift > 0:
            shift = np.random.uniform(-self.aug_intensity_shift, self.aug_intensity_shift)
            patch = np.clip(patch + shift, 0.0, 1.0)

        if self.aug_intensity_scale > 0:
            scale = np.random.uniform(
                1.0 - self.aug_intensity_scale, 1.0 + self.aug_intensity_scale
            )
            patch = np.clip(patch * scale, 0.0, 1.0)

        if self.aug_noise_std > 0:
            noise = np.random.normal(0, self.aug_noise_std, patch.shape).astype(np.float32)
            patch = patch + noise

        return patch

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.nodule_blocks:
            block = self._load_block(sample["annotation_id"])
            if block is None:
                patch = np.zeros(self.patch_size, dtype=np.float32)
            else:
                # Nodule blocks may differ in size from our patch_size;
                # crop/pad to match.
                patch = _crop_or_pad(block, self.patch_size)
        else:
            volume = self._load_volume(sample["seriesuid"])
            if volume is None:
                patch = np.zeros(self.patch_size, dtype=np.float32)
            else:
                center_voxel = (
                    int(round(sample["coord_z"])),
                    int(round(sample["coord_y"])),
                    int(round(sample["coord_x"])),
                )
                center_voxel = tuple(
                    max(0, min(c, s - 1)) for c, s in zip(center_voxel, volume.shape)
                )
                patch = extract_patch(volume, center_voxel, self.patch_size)

        if self.augment:
            patch = self._augment_patch(patch)

        patch_tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float()
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "patch": patch_tensor,
            "label": label_tensor,
            "seriesuid": sample["seriesuid"],
            "diameter_mm": sample["diameter_mm"],
        }


# ======================================================================
# Shared helpers (used by both dataset classes)
# ======================================================================

def _crop_or_pad(volume: np.ndarray, target_size: tuple) -> np.ndarray:
    """Crop or zero-pad a volume to *target_size*."""
    result = np.zeros(target_size, dtype=volume.dtype)
    slices_src = []
    slices_dst = []
    for i in range(3):
        if volume.shape[i] > target_size[i]:
            start = (volume.shape[i] - target_size[i]) // 2
            slices_src.append(slice(start, start + target_size[i]))
            slices_dst.append(slice(0, target_size[i]))
        else:
            start = (target_size[i] - volume.shape[i]) // 2
            slices_src.append(slice(0, volume.shape[i]))
            slices_dst.append(slice(start, start + volume.shape[i]))
    result[tuple(slices_dst)] = volume[tuple(slices_src)]
    return result


def _elastic_deformation(patch: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Apply random elastic deformation to a 3D patch."""
    shape = patch.shape
    dz = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="reflect")
    dy = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="reflect")
    dx = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode="reflect")
    z, y, x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    coords = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1),
    ]
    return ndimage.map_coordinates(patch, coords, order=1, mode="reflect").astype(patch.dtype)


# ======================================================================
# Combined multi-dataset wrapper
# ======================================================================

class CombinedLungDataset(ConcatDataset):
    """Concatenate multiple lung nodule datasets for joint training.

    Wraps ``torch.utils.data.ConcatDataset`` and logs the contribution
    of each constituent dataset.

    Usage::

        combined = CombinedLungDataset.from_config(config, split="train")
        loader = DataLoader(combined, ...)
    """

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str = "train",
        augment: bool | None = None,
    ) -> "CombinedLungDataset":
        """Build a combined dataset from the ``data.datasets`` config list.

        Falls back to a single LUNA16 dataset when ``data.datasets`` is not
        present (backwards-compatible).
        """
        data_config = config.get("data", {})
        datasets_cfg = data_config.get("datasets", None)

        if augment is None:
            augment = split == "train"

        # Legacy mode: single LUNA16 dataset
        if not datasets_cfg:
            ds = LUNA16Dataset(
                dataset_dir=data_config.get("dataset_dir", "./data/luna16"),
                config=config,
                split=split,
                val_split=data_config.get("val_split", 0.2),
                augment=augment,
                cache_dir=data_config.get("cache_dir", "./data/cache"),
            )
            return cls([ds])

        child_datasets: list[Dataset] = []
        for entry in datasets_cfg:
            kind = entry.get("type", "luna16")
            ds_dir = entry.get("dataset_dir")
            if ds_dir is None:
                continue
            enabled = entry.get("enabled", True)
            if not enabled:
                continue

            cache_dir = entry.get("cache_dir", data_config.get("cache_dir", "./data/cache"))
            val_split = entry.get("val_split", data_config.get("val_split", 0.2))

            if kind == "luna16":
                ds = LUNA16Dataset(
                    dataset_dir=ds_dir,
                    config=config,
                    split=split,
                    val_split=val_split,
                    augment=augment,
                    cache_dir=cache_dir,
                )
            elif kind == "luna25":
                ds = LUNA25Dataset(
                    dataset_dir=ds_dir,
                    config=config,
                    split=split,
                    val_split=val_split,
                    augment=augment,
                    cache_dir=cache_dir,
                )
            else:
                logger.warning("Unknown dataset type %r — skipping", kind)
                continue

            logger.info(
                "  %s (%s): %d samples",
                kind.upper(),
                split,
                len(ds),
            )
            child_datasets.append(ds)

        if not child_datasets:
            logger.warning("No datasets were loaded; training data will be empty")
            # Return an empty wrapper so the rest of the code doesn't crash
            return cls([LUNA16Dataset(
                dataset_dir=data_config.get("dataset_dir", "./data/luna16"),
                config=config,
                split=split,
                augment=augment,
            )])

        combined = cls(child_datasets)
        logger.info("Combined dataset (%s): %d total samples", split, len(combined))
        return combined
