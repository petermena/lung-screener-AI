"""Input validation for incoming DICOM studies.

Verifies that incoming data is actually a chest CT before running inference,
preventing nonsensical results on abdominal CTs, X-rays, or other modalities.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pydicom

logger = logging.getLogger(__name__)

# Expected DICOM attribute values for a chest CT
_VALID_MODALITIES = {"CT"}
_CHEST_BODY_PARTS = {
    "CHEST", "THORAX", "LUNG", "CHEST/ABDOMEN", "CHEST-ABDOMEN",
    "CHEST/ABD", "CHEST ABD", "CHESTABDPELVIS", "CHEST/ABDOMEN/PELVIS",
}
_MAX_SLICE_THICKNESS_MM = 5.0
_MIN_SLICES = 20


@dataclass
class ValidationResult:
    """Result of validating a DICOM study for chest CT processing."""

    is_valid: bool
    modality: str = ""
    body_part: str = ""
    slice_thickness: float = 0.0
    num_slices: int = 0
    warnings: list[str] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    def summary(self) -> str:
        lines = [f"Validation: {'PASS' if self.is_valid else 'FAIL'}"]
        lines.append(f"  Modality: {self.modality}")
        lines.append(f"  Body Part: {self.body_part}")
        lines.append(f"  Slice Thickness: {self.slice_thickness:.2f} mm")
        lines.append(f"  Number of Slices: {self.num_slices}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


def validate_dicom_series(dicom_dir: str | Path) -> ValidationResult:
    """Validate that a DICOM directory contains a suitable chest CT.

    Checks:
    - Modality is CT
    - Body part is chest/thorax (or absent with warning)
    - Slice thickness is reasonable (<= 5mm)
    - Sufficient number of slices (>= 20)
    - Consistent series UID across files

    Args:
        dicom_dir: Path to directory containing DICOM files.

    Returns:
        ValidationResult with pass/fail and details.
    """
    dicom_dir = Path(dicom_dir)
    errors = []
    warnings = []

    # Find DICOM files
    dcm_files = list(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        # Try without extension (some DICOM files have no extension)
        dcm_files = [
            f for f in dicom_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    if not dcm_files:
        return ValidationResult(
            is_valid=False,
            errors=["No DICOM files found in directory"],
        )

    # Read first file for metadata
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Cannot read DICOM file: {e}"],
        )

    # Check modality
    modality = getattr(ds, "Modality", "").upper()
    if modality not in _VALID_MODALITIES:
        errors.append(
            f"Expected CT modality, got '{modality}'. "
            "This tool is designed for CT chest scans only."
        )

    # Check body part
    body_part = getattr(ds, "BodyPartExamined", "").upper().strip()
    if not body_part:
        # Also check Study/Series description for chest keywords
        desc = " ".join([
            getattr(ds, "StudyDescription", ""),
            getattr(ds, "SeriesDescription", ""),
        ]).upper()
        if any(kw in desc for kw in ("CHEST", "THORAX", "LUNG", "LDCT")):
            body_part = "CHEST (from description)"
            warnings.append(
                "BodyPartExamined tag is empty; inferred CHEST from study/series description."
            )
        else:
            warnings.append(
                "BodyPartExamined tag is empty and description does not mention chest. "
                "Proceeding but results may be unreliable if this is not a chest CT."
            )
    elif body_part not in _CHEST_BODY_PARTS:
        errors.append(
            f"Body part '{body_part}' does not appear to be a chest CT. "
            "This tool is designed for chest CT screening."
        )

    # Check slice thickness
    slice_thickness = float(getattr(ds, "SliceThickness", 0.0))
    if slice_thickness <= 0:
        # Try SpacingBetweenSlices as fallback
        slice_thickness = float(getattr(ds, "SpacingBetweenSlices", 0.0))
    if slice_thickness <= 0:
        warnings.append(
            "Slice thickness not available in DICOM headers. "
            "Cannot verify reconstruction parameters."
        )
    elif slice_thickness > _MAX_SLICE_THICKNESS_MM:
        warnings.append(
            f"Slice thickness ({slice_thickness:.1f} mm) exceeds {_MAX_SLICE_THICKNESS_MM} mm. "
            "Thin-slice reconstruction (≤ 2.5 mm) is recommended for nodule detection. "
            "Results may have reduced sensitivity."
        )

    # Check number of slices
    num_slices = len(dcm_files)
    if num_slices < _MIN_SLICES:
        errors.append(
            f"Only {num_slices} slices found. "
            f"A complete chest CT typically has {_MIN_SLICES}+ slices."
        )

    # Check for consistent series UID
    series_uid = getattr(ds, "SeriesInstanceUID", "")
    if num_slices > 1:
        try:
            ds2 = pydicom.dcmread(str(dcm_files[-1]), stop_before_pixels=True)
            series_uid2 = getattr(ds2, "SeriesInstanceUID", "")
            if series_uid and series_uid2 and series_uid != series_uid2:
                warnings.append(
                    "Multiple series UIDs detected in directory. "
                    "Ensure only one CT series is present."
                )
        except Exception:
            pass

    # Check for scout/localizer
    image_type = getattr(ds, "ImageType", [])
    if isinstance(image_type, (list, pydicom.multival.MultiValue)):
        image_type_str = "\\".join(str(t) for t in image_type).upper()
    else:
        image_type_str = str(image_type).upper()
    if "LOCALIZER" in image_type_str or "SCOUT" in image_type_str:
        errors.append(
            "This appears to be a scout/localizer image, not an axial CT series."
        )

    # Check pixel spacing is present
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if pixel_spacing is None:
        warnings.append("PixelSpacing not found in DICOM headers.")

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        modality=modality,
        body_part=body_part or "(not specified)",
        slice_thickness=slice_thickness,
        num_slices=num_slices,
        warnings=warnings,
        errors=errors,
    )

    if is_valid:
        logger.info(f"DICOM validation passed: {modality}, {body_part}, {num_slices} slices")
    else:
        logger.warning(f"DICOM validation failed: {'; '.join(errors)}")

    return result


def validate_dicom_dataset(ds: pydicom.Dataset) -> ValidationResult:
    """Validate a single DICOM dataset object (e.g. from C-STORE).

    Lighter-weight check for real-time SCP validation.

    Args:
        ds: pydicom Dataset from an incoming DICOM object.

    Returns:
        ValidationResult.
    """
    errors = []
    warnings = []

    modality = getattr(ds, "Modality", "").upper()
    if modality not in _VALID_MODALITIES:
        errors.append(f"Expected CT modality, got '{modality}'")

    body_part = getattr(ds, "BodyPartExamined", "").upper().strip()
    if body_part and body_part not in _CHEST_BODY_PARTS:
        desc = " ".join([
            getattr(ds, "StudyDescription", ""),
            getattr(ds, "SeriesDescription", ""),
        ]).upper()
        if not any(kw in desc for kw in ("CHEST", "THORAX", "LUNG", "LDCT")):
            errors.append(f"Body part '{body_part}' is not a chest CT")

    image_type = getattr(ds, "ImageType", [])
    if isinstance(image_type, (list, pydicom.multival.MultiValue)):
        image_type_str = "\\".join(str(t) for t in image_type).upper()
    else:
        image_type_str = str(image_type).upper()
    if "LOCALIZER" in image_type_str:
        errors.append("Scout/localizer image, not axial CT")

    slice_thickness = float(getattr(ds, "SliceThickness", 0.0))

    return ValidationResult(
        is_valid=len(errors) == 0,
        modality=modality,
        body_part=body_part or "(not specified)",
        slice_thickness=slice_thickness,
        num_slices=1,
        warnings=warnings,
        errors=errors,
    )
