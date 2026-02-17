"""Inference engine for lung nodule detection.

Handles end-to-end prediction on new CT scans:
1. Preprocess the scan
2. Detect candidates
3. Classify each candidate
4. Determine nodule type (solid, part-solid, ground-glass)
5. Analyze calcification patterns (granuloma vs true nodule)
6. Apply non-maximum suppression
7. Generate structured findings with type-aware Lung-RADS
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast

from .calcification import (
    BENIGN_PATTERNS,
    CalcificationResult,
    analyze_calcification,
)
from .model import NODULE_TYPES, build_model
from .preprocessing import CTPreprocessor, extract_patch
from .risk_model import compute_lung_rads_with_risk

logger = logging.getLogger(__name__)


@dataclass
class NoduleFinding:
    """A detected nodule finding."""

    # Location in world coordinates (mm)
    x: float
    y: float
    z: float
    # Estimated diameter in mm
    diameter_mm: float
    # Model confidence (0-1)
    confidence: float
    # Lung-RADS category (computed from diameter and type)
    lung_rads: str = ""
    # Optional malignancy score (1-5 scale)
    malignancy_score: float = 0.0
    # Anatomical lobe (e.g. "right upper lobe")
    lobe: str = ""
    # Image/slice number in the series (1-based)
    image_number: int = 0
    # Series instance UID for this finding
    series_uid: str = ""
    # Nodule type: "solid", "part_solid", or "ground_glass"
    nodule_type: str = "solid"
    # Calcification analysis result (None if not analyzed)
    calcification: CalcificationResult | None = None

    def __post_init__(self):
        if not self.lung_rads:
            self.lung_rads = self._compute_lung_rads()
        # Apply Lung-RADS override for benign calcification
        if self.calcification and self.calcification.suggested_lung_rads_override:
            self.lung_rads = self.calcification.suggested_lung_rads_override

    def _compute_lung_rads(self) -> str:
        """Assign Lung-RADS category based on nodule diameter and type.

        Based on ACR Lung-RADS v2022 thresholds which vary by nodule type:
            Solid: 2 (<6mm), 3 (6-8mm), 4A (8-15mm), 4B (>=15mm)
            Part-solid: 2 (<6mm), 3 (6-8mm), 4A (8-15mm), 4B (>=15mm)
            Ground-glass: 2 (<30mm), 3 (>=30mm)
        """
        return compute_lung_rads_with_risk(self.diameter_mm, self.nodule_type)

    def to_dict(self) -> dict:
        result = {
            "location_mm": {"x": self.x, "y": self.y, "z": self.z},
            "diameter_mm": round(self.diameter_mm, 1),
            "confidence": round(self.confidence, 3),
            "lung_rads": self.lung_rads,
            "malignancy_score": round(self.malignancy_score, 2),
            "lobe": self.lobe,
            "image_number": self.image_number,
            "series_uid": self.series_uid,
            "nodule_type": self.nodule_type,
        }
        if self.calcification:
            result["calcification"] = self.calcification.to_dict()
        return result


def estimate_lobe(
    x: float, y: float, z: float,
    z_min: float, z_max: float,
) -> str:
    """Estimate which lung lobe a nodule is in from world coordinates.

    Uses the DICOM patient coordinate system:
        x: increases toward patient's left
        y: increases toward patient's posterior
        z: increases toward patient's superior (head)

    Approximation based on standard anatomical proportions:
        - Right lung (x < 0): upper / middle / lower lobes
        - Left lung  (x > 0): upper / lower lobes
        - The major fissure sits at roughly 40% of the z extent from the base
        - The minor fissure (right only) sits at roughly 65% of z extent

    Args:
        x, y, z: World coordinates in mm.
        z_min: Inferior extent of the lung volume (mm).
        z_max: Superior extent of the lung volume (mm).

    Returns:
        Lobe name, e.g. "right upper lobe".
    """
    z_range = z_max - z_min
    if z_range <= 0:
        return "indeterminate"

    # Normalized position: 0.0 = base (inferior), 1.0 = apex (superior)
    z_norm = (z - z_min) / z_range

    # Right lung: x < 0 in standard DICOM patient coords
    if x < 0:
        if z_norm >= 0.65:
            return "right upper lobe"
        elif z_norm >= 0.40:
            return "right middle lobe"
        else:
            return "right lower lobe"
    else:
        if z_norm >= 0.50:
            return "left upper lobe"
        else:
            return "left lower lobe"


def compute_image_number(
    z: float, origin_z: float, spacing_z: float,
) -> int:
    """Compute the 1-based image/slice number from a z world coordinate."""
    if spacing_z == 0:
        return 0
    return int(round((z - origin_z) / spacing_z)) + 1


@dataclass
class ScanResult:
    """Complete result for a processed CT scan."""

    series_uid: str = ""
    findings: list[NoduleFinding] = field(default_factory=list)
    lung_rads_overall: str = ""
    processing_status: str = "success"
    error_message: str = ""

    def __post_init__(self):
        if self.findings and not self.lung_rads_overall:
            # Overall Lung-RADS is the highest category among findings
            categories = [f.lung_rads for f in self.findings]
            self.lung_rads_overall = max(categories)

    def to_dict(self) -> dict:
        return {
            "series_uid": self.series_uid,
            "lung_rads_overall": self.lung_rads_overall or "1",
            "num_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "processing_status": self.processing_status,
            "error_message": self.error_message,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Lung Screening Result - Series: {self.series_uid}"]
        lines.append(f"Overall Lung-RADS: {self.lung_rads_overall or '1'}")
        lines.append(f"Findings: {len(self.findings)}")
        lines.append("")

        for i, f in enumerate(self.findings, 1):
            calc_info = ""
            if f.calcification and f.calcification.pattern.value != "none":
                calc_info = f", calcification={f.calcification.pattern.value}"
                if f.calcification.is_benign:
                    calc_info += " (BENIGN)"
            lines.append(
                f"  Finding {i}: {f.diameter_mm:.1f}mm {f.nodule_type} nodule at "
                f"({f.x:.1f}, {f.y:.1f}, {f.z:.1f})mm, "
                f"confidence={f.confidence:.1%}, "
                f"Lung-RADS {f.lung_rads}{calc_info}"
            )

        if not self.findings:
            lines.append("  No significant nodules detected.")

        return "\n".join(lines)

    def dictation(self) -> str:
        """Generate radiology dictation text ready to paste into reporting software.

        Produces prose in standard radiology report style with lobe location,
        series/image references, nodule type, Lung-RADS categorization, and
        ACR-aligned follow-up recommendations.
        """
        lines = []

        # Series reference header
        if self.series_uid:
            lines.append(f"Series: {self.series_uid}")
            lines.append("")

        lines.append("FINDINGS:")
        lines.append("")

        lung_rads = self.lung_rads_overall or "1"

        if not self.findings:
            lines.append(
                "No pulmonary nodules identified. "
                "The lungs are clear."
            )
            lines.append("")
            lines.append("IMPRESSION:")
            lines.append(
                "Lung-RADS Category 1: Negative. "
                "No pulmonary nodules. "
                "Continue annual screening with low-dose CT in 12 months."
            )
            return "\n".join(lines)

        lines.append("Pulmonary Nodules:")

        # Sort findings by size (largest first) for clinical relevance
        sorted_findings = sorted(
            self.findings, key=lambda f: f.diameter_mm, reverse=True
        )

        for i, f in enumerate(sorted_findings, 1):
            # Lobe location
            lobe_text = f.lobe.capitalize() if f.lobe else "indeterminate location"

            # Nodule type descriptor
            type_desc = _nodule_type_label(f.nodule_type)

            # Build the finding description
            nodule_desc = f"{i}. {lobe_text}: "
            nodule_desc += f"A {f.diameter_mm:.0f} mm {type_desc} pulmonary nodule"

            # Calcification description
            if f.calcification and f.calcification.is_benign:
                nodule_desc += (
                    f" with {f.calcification.pattern.value} calcification, "
                    "consistent with benign etiology"
                )
            elif f.calcification and f.calcification.pattern.value not in ("none", "partial"):
                nodule_desc += f" with {f.calcification.pattern.value} calcification"

            # Malignancy risk language
            if f.calcification and f.calcification.is_benign:
                pass  # Skip malignancy language for benign calcifications
            elif f.malignancy_score >= 4.0:
                nodule_desc += ", suspicious for malignancy"
            elif f.malignancy_score >= 3.0:
                nodule_desc += ", indeterminate"

            nodule_desc += f" (Lung-RADS {f.lung_rads})."

            # Series and image reference
            ref_parts = []
            if f.series_uid:
                ref_parts.append(f"Series {f.series_uid}")
            if f.image_number > 0:
                ref_parts.append(f"Image {f.image_number}")
            if ref_parts:
                nodule_desc += f" [{', '.join(ref_parts)}]"

            lines.append(nodule_desc)

        lines.append("")
        lines.append("IMPRESSION:")

        # Overall Lung-RADS with ACR-based recommendation
        rads_text = _lung_rads_impression(lung_rads, sorted_findings)
        lines.append(rads_text)

        lines.append("")
        lines.append(
            "Note: Computer-aided detection was used. "
            "Findings should be correlated with clinical history "
            "and prior imaging when available."
        )

        return "\n".join(lines)


def _nodule_type_label(nodule_type: str) -> str:
    """Convert nodule type code to radiology report language."""
    labels = {
        "solid": "solid",
        "part_solid": "part-solid",
        "ground_glass": "ground-glass",
    }
    return labels.get(nodule_type, "solid")


# Lung-RADS recommendation language per ACR guidelines
_LUNG_RADS_RECOMMENDATIONS = {
    "1": (
        "Negative",
        "No pulmonary nodules. Continue annual screening with low-dose CT in 12 months.",
    ),
    "2": (
        "Benign Appearance or Behavior",
        "Nodule(s) with very low likelihood of becoming a clinically active cancer. "
        "Continue annual screening with low-dose CT in 12 months.",
    ),
    "3": (
        "Probably Benign",
        "Probably benign finding(s). "
        "Short-term follow-up suggested. "
        "Recommend low-dose CT in 6 months.",
    ),
    "4A": (
        "Suspicious",
        "Findings suspicious for pulmonary malignancy. "
        "Recommend low-dose CT in 3 months, PET/CT may be considered.",
    ),
    "4B": (
        "Very Suspicious",
        "Findings very suspicious for pulmonary malignancy. "
        "Recommend tissue sampling and/or PET/CT. Consider multidisciplinary consultation.",
    ),
    "4X": (
        "Suspicious with Additional Features",
        "Category 3 or 4 finding with additional features suspicious for malignancy "
        "(e.g., spiculation, interval growth). "
        "Recommend tissue sampling and/or PET/CT. Consider multidisciplinary consultation.",
    ),
}


def _lung_rads_impression(category: str, findings: list) -> str:
    """Build the impression line for a given Lung-RADS category."""
    label, recommendation = _LUNG_RADS_RECOMMENDATIONS.get(
        category, ("Indeterminate", "Clinical correlation recommended.")
    )

    nodule_summary = ""
    if len(findings) == 1:
        f = findings[0]
        type_desc = _nodule_type_label(f.nodule_type)
        nodule_summary = f"A {f.diameter_mm:.0f} mm {type_desc} pulmonary nodule. "
    elif len(findings) > 1:
        descs = []
        for f in findings:
            type_desc = _nodule_type_label(f.nodule_type)
            descs.append(f"{f.diameter_mm:.0f} mm {type_desc}")
        nodule_summary = (
            f"{len(findings)} pulmonary nodules "
            f"measuring {', '.join(descs)}. "
        )

    return (
        f"Lung-RADS Category {category}: {label}. "
        f"{nodule_summary}"
        f"{recommendation}"
    )


class NoduleDetector:
    """End-to-end lung nodule detection engine."""

    def __init__(
        self,
        config: dict,
        model_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize preprocessor
        self.preprocessor = CTPreprocessor(config)

        # Inference config
        inf_config = config.get("inference", {})
        self.threshold = inf_config.get("threshold", 0.5)
        self.nms_distance_mm = inf_config.get("nms_distance_mm", 10.0)
        self.batch_size = inf_config.get("batch_size", 64)

        # Nodule type prediction
        self.predict_nodule_type = config.get("model", {}).get(
            "predict_nodule_type", False
        )

        # Build and load model
        self.model = build_model(config).to(self.device)
        if model_path:
            self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str | Path):
        """Load trained model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {path}")

    @torch.no_grad()
    def predict_scan(self, image: sitk.Image, series_uid: str = "") -> ScanResult:
        """Run full detection pipeline on a CT scan.

        Args:
            image: SimpleITK image of the CT scan.
            series_uid: Optional series identifier.

        Returns:
            ScanResult with all detected findings.
        """
        try:
            # Step 1: Preprocess
            processed = self.preprocessor.process_scan(image)
            volume = processed["volume"]
            volume_hu = processed["volume_hu"]
            candidates = processed["candidates"]
            spacing = processed["spacing"]
            origin = processed["origin"]

            logger.info(f"Found {len(candidates)} candidates in scan")

            # Compute z extent of the volume for lobe estimation
            vol_shape = volume.shape  # (z, y, x) in numpy order
            z_min = origin[2]  # SimpleITK origin z
            z_max = origin[2] + vol_shape[0] * spacing[2]

            if not candidates:
                return ScanResult(series_uid=series_uid)

            # Step 2: Extract patches for all candidates
            patch_size = tuple(
                self.config.get("model", {}).get("patch_size", [48, 48, 48])
            )
            patches = []
            for cand in candidates:
                patch = extract_patch(volume, cand["center_voxel"], patch_size)
                patches.append(patch)

            patches_array = np.stack(patches)[:, np.newaxis, ...]  # (N, 1, D, H, W)

            # Step 3: Classify in batches
            all_probs = []
            all_malignancy = []
            all_nodule_types = []
            for i in range(0, len(patches_array), self.batch_size):
                batch = torch.from_numpy(
                    patches_array[i : i + self.batch_size]
                ).float().to(self.device)

                with autocast():
                    output = self.model(batch)

                probs = torch.softmax(output["logits"], dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

                if "malignancy" in output:
                    all_malignancy.extend(output["malignancy"].cpu().numpy().flatten())

                if "nodule_type_logits" in output:
                    type_preds = torch.argmax(output["nodule_type_logits"], dim=1)
                    all_nodule_types.extend(type_preds.cpu().numpy())

            # Step 4: Filter by threshold
            findings = []
            for idx, (cand, prob) in enumerate(zip(candidates, all_probs)):
                if prob >= self.threshold:
                    # Convert voxel center to world coordinates
                    center_world = tuple(
                        o + c * s
                        for o, c, s in zip(origin, cand["center_voxel"], spacing)
                    )

                    malignancy = 0.0
                    if all_malignancy:
                        malignancy = float(all_malignancy[idx]) * 4.0 + 1.0  # Scale to [1, 5]

                    # Determine nodule type
                    nodule_type = "solid"
                    if all_nodule_types:
                        type_idx = int(all_nodule_types[idx])
                        if 0 <= type_idx < len(NODULE_TYPES):
                            nodule_type = NODULE_TYPES[type_idx]

                    wx = center_world[2]  # SimpleITK x
                    wy = center_world[1]  # SimpleITK y
                    wz = center_world[0]  # SimpleITK z

                    findings.append(NoduleFinding(
                        x=wx,
                        y=wy,
                        z=wz,
                        diameter_mm=cand["diameter_mm"],
                        confidence=float(prob),
                        malignancy_score=malignancy,
                        lobe=estimate_lobe(wx, wy, wz, z_min, z_max),
                        image_number=compute_image_number(wz, origin[2], spacing[2]),
                        series_uid=series_uid,
                        nodule_type=nodule_type,
                    ))

            # Step 5: Calcification analysis on raw HU volume
            for finding_idx, (finding, cand) in enumerate(
                zip(findings, [c for c, p in zip(candidates, all_probs) if p >= self.threshold])
            ):
                try:
                    calc_result = analyze_calcification(
                        volume_hu,
                        cand["center_voxel"],
                        cand["diameter_mm"],
                        spacing,
                    )
                    finding.calcification = calc_result
                    # Override Lung-RADS for benign calcification patterns
                    if calc_result.suggested_lung_rads_override:
                        finding.lung_rads = calc_result.suggested_lung_rads_override
                        logger.info(
                            f"Finding {finding_idx}: {calc_result.pattern.value} calcification "
                            f"→ Lung-RADS overridden to {calc_result.suggested_lung_rads_override}"
                        )
                except Exception as e:
                    logger.warning(f"Calcification analysis failed for finding {finding_idx}: {e}")

            # Step 6: Non-maximum suppression
            findings = self._nms(findings)

            logger.info(f"Detected {len(findings)} nodules after NMS")

            return ScanResult(series_uid=series_uid, findings=findings)

        except Exception as e:
            logger.error(f"Error processing scan: {e}")
            return ScanResult(
                series_uid=series_uid,
                processing_status="error",
                error_message=str(e),
            )

    def _nms(self, findings: list[NoduleFinding]) -> list[NoduleFinding]:
        """Non-maximum suppression based on distance and confidence.

        If two findings are within nms_distance_mm of each other,
        keep only the one with higher confidence.
        """
        if len(findings) <= 1:
            return findings

        # Sort by confidence (descending)
        findings = sorted(findings, key=lambda f: f.confidence, reverse=True)
        keep = []

        for finding in findings:
            # Check if this finding is too close to any already-kept finding
            too_close = False
            for kept in keep:
                dist = np.sqrt(
                    (finding.x - kept.x) ** 2
                    + (finding.y - kept.y) ** 2
                    + (finding.z - kept.z) ** 2
                )
                if dist < self.nms_distance_mm:
                    too_close = True
                    break

            if not too_close:
                keep.append(finding)

        return keep
