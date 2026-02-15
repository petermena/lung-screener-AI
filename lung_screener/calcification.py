"""Calcification detection for distinguishing benign granulomas from nodules.

Analyzes Hounsfield Unit density patterns within candidate nodule regions
to detect calcification. Specific calcification patterns are considered
definitively benign per ACR Lung-RADS and should be downgraded accordingly.

Benign calcification patterns (Lung-RADS → Category 1 or 2):
- Complete/diffuse calcification: entire nodule is calcified
- Central calcification: dense core with soft-tissue rim (granuloma)
- Popcorn calcification: scattered chunky calcifications (hamartoma)
- Laminated/concentric calcification: ring-like layers

Suspicious calcification patterns (do NOT downgrade):
- Eccentric calcification: off-center focus (may indicate malignancy)
- Stippled/punctate: tiny scattered foci within soft-tissue mass
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# HU thresholds for calcification analysis
_CALCIFICATION_HU_THRESHOLD = 200  # Voxels above this are likely calcified
_DENSE_CALCIFICATION_HU = 400  # High-confidence calcification
_BONE_HU_THRESHOLD = 700  # Very dense; cortical bone / dense calcium
_SOFT_TISSUE_HU_RANGE = (-100, 100)  # Typical soft-tissue density
_FAT_HU_RANGE = (-150, -50)  # Fat density (lipid-rich hamartoma clue)


class CalcificationPattern(Enum):
    """Calcification pattern classification."""

    NONE = "none"
    COMPLETE = "complete"  # Entirely calcified → benign
    CENTRAL = "central"  # Dense center, soft rim → granuloma
    POPCORN = "popcorn"  # Scattered chunks → hamartoma
    LAMINATED = "laminated"  # Concentric rings → benign
    ECCENTRIC = "eccentric"  # Off-center focus → suspicious
    PUNCTATE = "punctate"  # Tiny scattered foci → indeterminate
    PARTIAL = "partial"  # Some calcification, pattern unclear


# Patterns that are definitively benign per ACR guidelines
BENIGN_PATTERNS = {
    CalcificationPattern.COMPLETE,
    CalcificationPattern.CENTRAL,
    CalcificationPattern.POPCORN,
    CalcificationPattern.LAMINATED,
}

# Patterns that should NOT be downgraded
SUSPICIOUS_PATTERNS = {
    CalcificationPattern.ECCENTRIC,
    CalcificationPattern.PUNCTATE,
}


@dataclass
class CalcificationResult:
    """Result of calcification analysis for a single candidate."""

    pattern: CalcificationPattern
    calcification_fraction: float  # 0-1, fraction of voxels above threshold
    mean_hu: float  # Mean HU of the candidate region
    max_hu: float  # Maximum HU in the candidate
    is_benign: bool  # True if pattern is definitively benign
    description: str = ""
    suggested_lung_rads_override: str = ""  # "" = no override

    def __post_init__(self):
        self.is_benign = self.pattern in BENIGN_PATTERNS
        if not self.description:
            self.description = _pattern_description(self.pattern)
        if not self.suggested_lung_rads_override and self.is_benign:
            if self.pattern == CalcificationPattern.COMPLETE:
                self.suggested_lung_rads_override = "1"
            else:
                self.suggested_lung_rads_override = "2"

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern.value,
            "calcification_fraction": round(self.calcification_fraction, 3),
            "mean_hu": round(self.mean_hu, 1),
            "max_hu": round(self.max_hu, 1),
            "is_benign": self.is_benign,
            "description": self.description,
            "suggested_lung_rads_override": self.suggested_lung_rads_override,
        }


def analyze_calcification(
    volume_hu: np.ndarray,
    center_voxel: tuple[int, int, int],
    diameter_mm: float,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CalcificationResult:
    """Analyze calcification within a candidate nodule region.

    Extracts the candidate region from the raw HU volume (before
    windowing/normalization) and classifies the calcification pattern.

    Args:
        volume_hu: Raw CT volume in Hounsfield Units (NOT normalized).
        center_voxel: (z, y, x) center of the candidate.
        diameter_mm: Estimated nodule diameter in mm.
        spacing: Voxel spacing in mm.

    Returns:
        CalcificationResult with pattern classification.
    """
    spacing = np.array(spacing)

    # Extract a spherical region around the candidate
    radius_mm = max(diameter_mm / 2.0, 2.0)  # At least 2mm radius
    radius_voxels = radius_mm / spacing

    # Build a spherical mask around the center
    cz, cy, cx = center_voxel
    vol_shape = np.array(volume_hu.shape)

    # Bounding box for the region
    r_int = np.ceil(radius_voxels).astype(int) + 1
    z_lo = max(0, cz - r_int[0])
    z_hi = min(vol_shape[0], cz + r_int[0] + 1)
    y_lo = max(0, cy - r_int[1])
    y_hi = min(vol_shape[1], cy + r_int[1] + 1)
    x_lo = max(0, cx - r_int[2])
    x_hi = min(vol_shape[2], cx + r_int[2] + 1)

    region_hu = volume_hu[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    if region_hu.size == 0:
        return CalcificationResult(
            pattern=CalcificationPattern.NONE,
            calcification_fraction=0.0,
            mean_hu=0.0,
            max_hu=0.0,
        )

    # Create spherical mask within the bounding box
    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_mm = np.sqrt(
        ((zz - cz) * spacing[0]) ** 2
        + ((yy - cy) * spacing[1]) ** 2
        + ((xx - cx) * spacing[2]) ** 2
    )
    sphere_mask = dist_mm <= radius_mm

    # Get HU values within the nodule region
    nodule_hu = region_hu[sphere_mask]
    if nodule_hu.size == 0:
        return CalcificationResult(
            pattern=CalcificationPattern.NONE,
            calcification_fraction=0.0,
            mean_hu=0.0,
            max_hu=0.0,
        )

    mean_hu = float(np.mean(nodule_hu))
    max_hu = float(np.max(nodule_hu))
    min_hu = float(np.min(nodule_hu))

    # Fraction of voxels above calcification threshold
    calc_voxels = nodule_hu > _CALCIFICATION_HU_THRESHOLD
    calc_fraction = float(np.mean(calc_voxels))

    dense_calc_voxels = nodule_hu > _DENSE_CALCIFICATION_HU
    dense_fraction = float(np.mean(dense_calc_voxels))

    # --- Pattern classification ---

    # No significant calcification
    if calc_fraction < 0.05 and max_hu < _CALCIFICATION_HU_THRESHOLD:
        return CalcificationResult(
            pattern=CalcificationPattern.NONE,
            calcification_fraction=calc_fraction,
            mean_hu=mean_hu,
            max_hu=max_hu,
        )

    # COMPLETE calcification: nearly all voxels are calcified
    if calc_fraction > 0.80 and mean_hu > _CALCIFICATION_HU_THRESHOLD:
        return CalcificationResult(
            pattern=CalcificationPattern.COMPLETE,
            calcification_fraction=calc_fraction,
            mean_hu=mean_hu,
            max_hu=max_hu,
        )

    # For spatial pattern analysis, we need the 3D calcification mask
    calc_mask_3d = (region_hu > _CALCIFICATION_HU_THRESHOLD) & sphere_mask
    soft_mask_3d = (
        (region_hu > _SOFT_TISSUE_HU_RANGE[0])
        & (region_hu < _SOFT_TISSUE_HU_RANGE[1])
        & sphere_mask
    )

    # CENTRAL calcification: dense core surrounded by soft tissue
    if calc_fraction > 0.15:
        pattern = _check_central_pattern(
            calc_mask_3d, sphere_mask, center_voxel,
            (z_lo, y_lo, x_lo), spacing, radius_mm,
        )
        if pattern == CalcificationPattern.CENTRAL:
            return CalcificationResult(
                pattern=CalcificationPattern.CENTRAL,
                calcification_fraction=calc_fraction,
                mean_hu=mean_hu,
                max_hu=max_hu,
            )

    # POPCORN calcification: multiple discrete calcified foci
    if 0.10 < calc_fraction < 0.70:
        pattern = _check_popcorn_pattern(calc_mask_3d)
        if pattern == CalcificationPattern.POPCORN:
            return CalcificationResult(
                pattern=CalcificationPattern.POPCORN,
                calcification_fraction=calc_fraction,
                mean_hu=mean_hu,
                max_hu=max_hu,
            )

    # LAMINATED calcification: concentric ring pattern
    if 0.15 < calc_fraction < 0.60:
        pattern = _check_laminated_pattern(
            region_hu, sphere_mask, center_voxel,
            (z_lo, y_lo, x_lo), spacing, radius_mm,
        )
        if pattern == CalcificationPattern.LAMINATED:
            return CalcificationResult(
                pattern=CalcificationPattern.LAMINATED,
                calcification_fraction=calc_fraction,
                mean_hu=mean_hu,
                max_hu=max_hu,
            )

    # ECCENTRIC calcification: single off-center focus
    if 0.05 < calc_fraction < 0.40:
        pattern = _check_eccentric_pattern(
            calc_mask_3d, sphere_mask, center_voxel,
            (z_lo, y_lo, x_lo), spacing, radius_mm,
        )
        if pattern == CalcificationPattern.ECCENTRIC:
            return CalcificationResult(
                pattern=CalcificationPattern.ECCENTRIC,
                calcification_fraction=calc_fraction,
                mean_hu=mean_hu,
                max_hu=max_hu,
            )

    # PUNCTATE: very small scattered foci
    if calc_fraction < 0.15 and max_hu > _DENSE_CALCIFICATION_HU:
        return CalcificationResult(
            pattern=CalcificationPattern.PUNCTATE,
            calcification_fraction=calc_fraction,
            mean_hu=mean_hu,
            max_hu=max_hu,
        )

    # Partial calcification with no clear pattern
    if calc_fraction > 0.05:
        return CalcificationResult(
            pattern=CalcificationPattern.PARTIAL,
            calcification_fraction=calc_fraction,
            mean_hu=mean_hu,
            max_hu=max_hu,
        )

    return CalcificationResult(
        pattern=CalcificationPattern.NONE,
        calcification_fraction=calc_fraction,
        mean_hu=mean_hu,
        max_hu=max_hu,
    )


def _check_central_pattern(
    calc_mask: np.ndarray,
    nodule_mask: np.ndarray,
    center_voxel: tuple[int, int, int],
    bbox_origin: tuple[int, int, int],
    spacing: np.ndarray,
    radius_mm: float,
) -> CalcificationPattern:
    """Check if calcification follows a central (granuloma) pattern.

    Central calcification: the calcified voxels are concentrated in
    the inner 50% of the nodule radius, with soft tissue in the outer shell.
    """
    cz, cy, cx = center_voxel
    zo, yo, xo = bbox_origin
    shape = calc_mask.shape

    zz, yy, xx = np.mgrid[zo:zo + shape[0], yo:yo + shape[1], xo:xo + shape[2]]
    dist_mm = np.sqrt(
        ((zz - cz) * spacing[0]) ** 2
        + ((yy - cy) * spacing[1]) ** 2
        + ((xx - cx) * spacing[2]) ** 2
    )

    inner_mask = (dist_mm <= radius_mm * 0.5) & nodule_mask
    outer_mask = (dist_mm > radius_mm * 0.5) & (dist_mm <= radius_mm) & nodule_mask

    inner_count = inner_mask.sum()
    outer_count = outer_mask.sum()

    if inner_count == 0 or outer_count == 0:
        return CalcificationPattern.PARTIAL

    inner_calc_frac = (calc_mask & inner_mask).sum() / inner_count
    outer_calc_frac = (calc_mask & outer_mask).sum() / outer_count

    # Central pattern: inner region mostly calcified, outer mostly not
    if inner_calc_frac > 0.5 and outer_calc_frac < 0.2:
        return CalcificationPattern.CENTRAL

    return CalcificationPattern.PARTIAL


def _check_popcorn_pattern(calc_mask: np.ndarray) -> CalcificationPattern:
    """Check for popcorn calcification (hamartoma pattern).

    Popcorn pattern: multiple discrete, separated calcified foci
    of varying size, none of which dominates the nodule.
    """
    labeled = ndimage.label(calc_mask)[0]
    num_components = labeled.max()

    # Popcorn requires multiple discrete foci (≥3)
    if num_components >= 3:
        sizes = ndimage.sum(calc_mask, labeled, range(1, num_components + 1))
        sizes = np.array(sizes)
        total_calc = sizes.sum()

        if total_calc > 0:
            # No single component should dominate (>60% of total calcification)
            max_frac = sizes.max() / total_calc
            if max_frac < 0.60:
                return CalcificationPattern.POPCORN

    return CalcificationPattern.PARTIAL


def _check_laminated_pattern(
    region_hu: np.ndarray,
    nodule_mask: np.ndarray,
    center_voxel: tuple[int, int, int],
    bbox_origin: tuple[int, int, int],
    spacing: np.ndarray,
    radius_mm: float,
) -> CalcificationPattern:
    """Check for laminated/concentric calcification pattern.

    Laminated pattern: alternating shells of calcified and non-calcified
    tissue radiating from center, creating a ring-like appearance.
    """
    cz, cy, cx = center_voxel
    zo, yo, xo = bbox_origin
    shape = region_hu.shape

    zz, yy, xx = np.mgrid[zo:zo + shape[0], yo:yo + shape[1], xo:xo + shape[2]]
    dist_mm = np.sqrt(
        ((zz - cz) * spacing[0]) ** 2
        + ((yy - cy) * spacing[1]) ** 2
        + ((xx - cx) * spacing[2]) ** 2
    )

    # Sample mean HU in radial shells
    n_shells = 5
    shell_boundaries = np.linspace(0, radius_mm, n_shells + 1)
    shell_means = []

    for i in range(n_shells):
        shell_mask = (
            (dist_mm >= shell_boundaries[i])
            & (dist_mm < shell_boundaries[i + 1])
            & nodule_mask
        )
        if shell_mask.sum() > 0:
            shell_means.append(float(region_hu[shell_mask].mean()))
        else:
            shell_means.append(0.0)

    # Laminated pattern: alternating high/low density shells
    if len(shell_means) >= 4:
        transitions = 0
        for i in range(1, len(shell_means)):
            above_prev = shell_means[i - 1] > _CALCIFICATION_HU_THRESHOLD
            above_curr = shell_means[i] > _CALCIFICATION_HU_THRESHOLD
            if above_prev != above_curr:
                transitions += 1

        # At least 2 transitions (calcified → not → calcified)
        if transitions >= 2:
            return CalcificationPattern.LAMINATED

    return CalcificationPattern.PARTIAL


def _check_eccentric_pattern(
    calc_mask: np.ndarray,
    nodule_mask: np.ndarray,
    center_voxel: tuple[int, int, int],
    bbox_origin: tuple[int, int, int],
    spacing: np.ndarray,
    radius_mm: float,
) -> CalcificationPattern:
    """Check for eccentric calcification (suspicious pattern).

    Eccentric pattern: a single focus of calcification that is clearly
    off-center, often indicating engulfment of a pre-existing calcification
    by a growing malignant mass.
    """
    labeled, num_components = ndimage.label(calc_mask)

    if num_components == 0:
        return CalcificationPattern.PARTIAL

    # Find the largest calcified component
    sizes = ndimage.sum(calc_mask, labeled, range(1, num_components + 1))
    largest_label = np.argmax(sizes) + 1

    # Find its centroid
    largest_mask = labeled == largest_label
    calc_centroid = ndimage.center_of_mass(largest_mask)

    cz, cy, cx = center_voxel
    zo, yo, xo = bbox_origin

    # Distance from calcification centroid to nodule center
    calc_cz = calc_centroid[0] + zo
    calc_cy = calc_centroid[1] + yo
    calc_cx = calc_centroid[2] + xo

    offset_mm = np.sqrt(
        ((calc_cz - cz) * spacing[0]) ** 2
        + ((calc_cy - cy) * spacing[1]) ** 2
        + ((calc_cx - cx) * spacing[2]) ** 2
    )

    # Eccentric if the calcification center is >40% of radius from nodule center
    # and there's only 1-2 calcification foci
    if offset_mm > radius_mm * 0.4 and num_components <= 2:
        return CalcificationPattern.ECCENTRIC

    return CalcificationPattern.PARTIAL


def _pattern_description(pattern: CalcificationPattern) -> str:
    """Human-readable description of calcification pattern."""
    descriptions = {
        CalcificationPattern.NONE: "No calcification detected",
        CalcificationPattern.COMPLETE: (
            "Complete/diffuse calcification. Consistent with calcified granuloma. "
            "Definitively benign per ACR Lung-RADS."
        ),
        CalcificationPattern.CENTRAL: (
            "Central calcification pattern. Consistent with granuloma. "
            "Definitively benign per ACR Lung-RADS."
        ),
        CalcificationPattern.POPCORN: (
            "Popcorn calcification pattern. Consistent with hamartoma. "
            "Definitively benign per ACR Lung-RADS."
        ),
        CalcificationPattern.LAMINATED: (
            "Laminated/concentric calcification pattern. "
            "Consistent with granuloma. Definitively benign per ACR Lung-RADS."
        ),
        CalcificationPattern.ECCENTRIC: (
            "Eccentric calcification pattern. May represent malignancy "
            "engulfing a pre-existing granuloma. Does NOT indicate benignity."
        ),
        CalcificationPattern.PUNCTATE: (
            "Punctate calcification. Small scattered foci that are "
            "indeterminate and do not indicate benignity."
        ),
        CalcificationPattern.PARTIAL: (
            "Partial calcification with no clearly benign pattern. "
            "Further evaluation recommended."
        ),
    }
    return descriptions.get(pattern, "Unknown calcification pattern")
