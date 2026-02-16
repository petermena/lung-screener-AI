#!/usr/bin/env python3
"""Generate synthetic LUNA16-format data for pipeline validation.

Creates fake CT volumes with embedded spherical "nodules" and matching
annotation/candidate CSV files. This allows the full training pipeline
to be tested end-to-end without downloading the real ~100GB dataset.

The synthetic data mimics the LUNA16 directory structure:
    data/luna16/
        annotations.csv
        candidates_V2.csv
        subset0/
            <seriesuid>.mhd
            <seriesuid>.raw
        ...

Usage:
    python scripts/generate_synthetic_data.py [--num-scans 20] [--output-dir ./data/luna16]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def generate_synthetic_volume(
    shape=(128, 128, 128),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    num_nodules=0,
    nodule_positions=None,
    nodule_diameters=None,
    rng=None,
):
    """Generate a synthetic CT volume with optional embedded nodules.

    The volume simulates:
    - Background tissue (~40 HU)
    - Lung parenchyma (air-filled, ~-700 HU)
    - Nodules (dense tissue, ~50-100 HU)
    - Some random noise for realism

    Returns:
        sitk_image: SimpleITK image with HU values
        nodule_info: list of dicts with world-coordinate nodule locations
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create base volume with tissue-like HU values
    volume = np.full(shape, 40.0, dtype=np.float32)  # soft tissue

    # Create ellipsoidal lung regions (left and right)
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center_z, center_y = shape[0] // 2, shape[1] // 2

    # Left lung
    left_cx = shape[2] // 4
    left_mask = (
        ((z - center_z) / (shape[0] * 0.4)) ** 2
        + ((y - center_y) / (shape[1] * 0.35)) ** 2
        + ((x - left_cx) / (shape[2] * 0.2)) ** 2
    ) < 1.0

    # Right lung
    right_cx = 3 * shape[2] // 4
    right_mask = (
        ((z - center_z) / (shape[0] * 0.4)) ** 2
        + ((y - center_y) / (shape[1] * 0.35)) ** 2
        + ((x - right_cx) / (shape[2] * 0.2)) ** 2
    ) < 1.0

    lung_mask = left_mask | right_mask
    volume[lung_mask] = -700.0  # lung parenchyma

    # Add noise
    volume += rng.normal(0, 15, shape).astype(np.float32)

    # Embed nodules
    nodule_info = []
    if nodule_positions is None and num_nodules > 0:
        nodule_positions = []
        nodule_diameters = []
        for _ in range(num_nodules):
            # Place nodules inside lung regions
            lung_coords = np.argwhere(lung_mask)
            idx = rng.integers(0, len(lung_coords))
            pos = lung_coords[idx]
            diameter = rng.uniform(4.0, 15.0)  # 4-15mm
            nodule_positions.append(pos)
            nodule_diameters.append(diameter)

    if nodule_positions is not None:
        for pos, diameter in zip(nodule_positions, nodule_diameters):
            radius_voxels = diameter / (2.0 * spacing[0])  # assuming isotropic
            pz, py, px = pos
            dist = np.sqrt(
                (z - pz) ** 2 + (y - py) ** 2 + (x - px) ** 2
            )
            nodule_mask = dist < radius_voxels
            # Nodule HU: 50-100 (dense tissue)
            volume[nodule_mask] = rng.uniform(50, 100)

            # Convert voxel to world coordinates
            world_x = float(px * spacing[2] + origin[2])
            world_y = float(py * spacing[1] + origin[1])
            world_z = float(pz * spacing[0] + origin[0])

            nodule_info.append({
                "coordX": world_x,
                "coordY": world_y,
                "coordZ": world_z,
                "diameter_mm": float(diameter),
                "center_voxel": (int(pz), int(py), int(px)),
            })

    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)

    return sitk_image, nodule_info


def generate_negative_candidates(
    shape, lung_mask_shape, spacing, origin, num_candidates, nodule_positions, rng
):
    """Generate false-positive candidate locations (no real nodule)."""
    candidates = []
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    center_z, center_y = shape[0] // 2, shape[1] // 2
    left_cx = shape[2] // 4
    right_cx = 3 * shape[2] // 4

    left_mask = (
        ((z - center_z) / (shape[0] * 0.4)) ** 2
        + ((y - center_y) / (shape[1] * 0.35)) ** 2
        + ((x - left_cx) / (shape[2] * 0.2)) ** 2
    ) < 1.0
    right_mask = (
        ((z - center_z) / (shape[0] * 0.4)) ** 2
        + ((y - center_y) / (shape[1] * 0.35)) ** 2
        + ((x - right_cx) / (shape[2] * 0.2)) ** 2
    ) < 1.0
    lung_mask = left_mask | right_mask
    lung_coords = np.argwhere(lung_mask)

    if len(lung_coords) == 0:
        return candidates

    for _ in range(num_candidates):
        # Pick a random lung voxel far from any real nodule
        for _attempt in range(50):
            idx = rng.integers(0, len(lung_coords))
            pos = lung_coords[idx]
            # Check distance from all nodules
            far_enough = True
            for npos in nodule_positions:
                dist = np.sqrt(np.sum((pos - np.array(npos)) ** 2))
                if dist < 15:  # at least 15 voxels away
                    far_enough = False
                    break
            if far_enough:
                break

        world_x = float(pos[2] * spacing[2] + origin[2])
        world_y = float(pos[1] * spacing[1] + origin[1])
        world_z = float(pos[0] * spacing[0] + origin[0])
        candidates.append({
            "coordX": world_x,
            "coordY": world_y,
            "coordZ": world_z,
            "class": 0,
        })

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LUNA16 data")
    parser.add_argument("--num-scans", type=int, default=20, help="Number of synthetic scans")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic", help="Output directory (default: ./data/synthetic to avoid overwriting real data)")
    parser.add_argument("--volume-size", type=int, default=128, help="Volume size (cubic)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shape = (args.volume_size, args.volume_size, args.volume_size)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    all_annotations = []
    all_candidates = []

    print(f"Generating {args.num_scans} synthetic scans in {output_dir}/")

    for i in range(args.num_scans):
        # Distribute across subsets (put in subset0 for simplicity)
        subset_idx = i % 10
        subset_dir = output_dir / f"subset{subset_idx}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique series UID
        seriesuid = f"1.3.6.1.4.1.14519.5.2.1.0.0.{1000 + i}"

        # Randomly decide number of nodules (0-3)
        num_nodules = rng.integers(0, 4)

        image, nodule_info = generate_synthetic_volume(
            shape=shape,
            spacing=spacing,
            origin=origin,
            num_nodules=num_nodules,
            rng=rng,
        )

        # Save as .mhd/.raw
        mhd_path = subset_dir / f"{seriesuid}.mhd"
        sitk.WriteImage(image, str(mhd_path))

        # Record annotations (positive nodules)
        nodule_positions = []
        for nod in nodule_info:
            all_annotations.append({
                "seriesuid": seriesuid,
                "coordX": nod["coordX"],
                "coordY": nod["coordY"],
                "coordZ": nod["coordZ"],
                "diameter_mm": nod["diameter_mm"],
            })
            all_candidates.append({
                "seriesuid": seriesuid,
                "coordX": nod["coordX"],
                "coordY": nod["coordY"],
                "coordZ": nod["coordZ"],
                "class": 1,
            })
            nodule_positions.append(nod["center_voxel"])

        # Generate negative candidates
        num_neg = rng.integers(3, 8)
        neg_candidates = generate_negative_candidates(
            shape, shape, spacing, origin, num_neg, nodule_positions, rng
        )
        for nc in neg_candidates:
            nc["seriesuid"] = seriesuid
            all_candidates.append(nc)

        status = f"  Scan {i+1:3d}/{args.num_scans}: {seriesuid} — {num_nodules} nodules, {len(neg_candidates)} neg candidates"
        print(status)

    # Write annotations.csv (back up existing file first)
    ann_path = output_dir / "annotations.csv"
    if ann_path.exists():
        backup = ann_path.with_suffix(".csv.real_backup")
        if not backup.exists():
            import shutil
            shutil.copy2(ann_path, backup)
            print(f"  Backed up existing {ann_path} -> {backup}")
    with open(ann_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        writer.writeheader()
        writer.writerows(all_annotations)

    # Write candidates_V2.csv (back up existing file first)
    cand_path = output_dir / "candidates_V2.csv"
    if cand_path.exists():
        backup = cand_path.with_suffix(".csv.real_backup")
        if not backup.exists():
            import shutil
            shutil.copy2(cand_path, backup)
            print(f"  Backed up existing {cand_path} -> {backup}")
    with open(cand_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "class"])
        writer.writeheader()
        writer.writerows(all_candidates)

    # Summary
    total_pos = sum(1 for c in all_candidates if c["class"] == 1)
    total_neg = sum(1 for c in all_candidates if c["class"] == 0)
    print(f"\nDone!")
    print(f"  Annotations: {len(all_annotations)} nodules in {ann_path}")
    print(f"  Candidates:  {len(all_candidates)} ({total_pos} pos, {total_neg} neg) in {cand_path}")
    print(f"  Volumes:     {args.num_scans} .mhd/.raw pairs across subset directories")
    print(f"\nTo train:")
    print(f"  lung-screener train --checkpoint-dir ./checkpoints")


if __name__ == "__main__":
    main()
