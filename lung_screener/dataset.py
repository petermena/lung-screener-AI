"""Dataset classes for LUNA16/LIDC-IDRI lung nodule data.

Handles loading annotations, creating train/val splits, and providing
3D patches with data augmentation for training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from .preprocessing import CTPreprocessor, extract_patch, load_mhd


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
                volume = np.load(cache_path)
                self._volume_cache[seriesuid] = volume
                return volume

        # Load from raw data
        mhd_path = self._find_volume_path(seriesuid)
        if mhd_path is None:
            return None

        image = load_mhd(mhd_path)
        result = self.preprocessor.process_scan(image)
        volume = result["volume"]

        # Save to disk cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{seriesuid}.npy", volume)

        # Keep in memory cache (limit to ~10 volumes to avoid OOM)
        if len(self._volume_cache) < 10:
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
        """Apply random data augmentation to a 3D patch."""
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

        # Random Gaussian noise
        if self.aug_noise_std > 0:
            noise = np.random.normal(0, self.aug_noise_std, patch.shape).astype(np.float32)
            patch = patch + noise

        return patch

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
