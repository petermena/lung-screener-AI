"""Inference engine for lung nodule detection.

Handles end-to-end prediction on new CT scans:
1. Preprocess the scan
2. Detect candidates
3. Classify each candidate
4. Apply non-maximum suppression
5. Generate structured findings
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast

from .model import build_model
from .preprocessing import CTPreprocessor, extract_patch

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
    # Lung-RADS category (computed from diameter)
    lung_rads: str = ""
    # Optional malignancy score (1-5 scale)
    malignancy_score: float = 0.0

    def __post_init__(self):
        if not self.lung_rads:
            self.lung_rads = self._compute_lung_rads()

    def _compute_lung_rads(self) -> str:
        """Assign Lung-RADS category based on nodule diameter.

        Based on ACR Lung-RADS v2022 for solid nodules:
            1: No nodules or clearly benign
            2: <6mm solid nodule
            3: 6-8mm solid nodule
            4A: 8-15mm solid nodule
            4B: >=15mm solid nodule
        """
        d = self.diameter_mm
        if d < 6:
            return "2"
        elif d < 8:
            return "3"
        elif d < 15:
            return "4A"
        else:
            return "4B"

    def to_dict(self) -> dict:
        return {
            "location_mm": {"x": self.x, "y": self.y, "z": self.z},
            "diameter_mm": round(self.diameter_mm, 1),
            "confidence": round(self.confidence, 3),
            "lung_rads": self.lung_rads,
            "malignancy_score": round(self.malignancy_score, 2),
        }


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
            lines.append(
                f"  Finding {i}: {f.diameter_mm:.1f}mm nodule at "
                f"({f.x:.1f}, {f.y:.1f}, {f.z:.1f})mm, "
                f"confidence={f.confidence:.1%}, "
                f"Lung-RADS {f.lung_rads}"
            )

        if not self.findings:
            lines.append("  No significant nodules detected.")

        return "\n".join(lines)

    def dictation(self) -> str:
        """Generate radiology dictation text ready to paste into reporting software.

        Produces prose in standard radiology report style with Lung-RADS
        categorization and ACR-aligned follow-up recommendations.
        """
        lines = []
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
            # Describe the nodule
            nodule_desc = f"{i}. "
            nodule_desc += f"A {f.diameter_mm:.0f} mm solid pulmonary nodule"

            # Malignancy risk language
            if f.malignancy_score >= 4.0:
                nodule_desc += ", suspicious for malignancy"
            elif f.malignancy_score >= 3.0:
                nodule_desc += ", indeterminate"

            nodule_desc += f" (Lung-RADS {f.lung_rads})."

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
}


def _lung_rads_impression(category: str, findings: list) -> str:
    """Build the impression line for a given Lung-RADS category."""
    label, recommendation = _LUNG_RADS_RECOMMENDATIONS.get(
        category, ("Indeterminate", "Clinical correlation recommended.")
    )

    nodule_summary = ""
    if len(findings) == 1:
        f = findings[0]
        nodule_summary = f"A {f.diameter_mm:.0f} mm pulmonary nodule. "
    elif len(findings) > 1:
        sizes = ", ".join(f"{f.diameter_mm:.0f} mm" for f in findings)
        nodule_summary = (
            f"{len(findings)} pulmonary nodules "
            f"measuring {sizes}. "
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
            candidates = processed["candidates"]
            spacing = processed["spacing"]
            origin = processed["origin"]

            logger.info(f"Found {len(candidates)} candidates in scan")

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

                    findings.append(NoduleFinding(
                        x=center_world[2],  # SimpleITK x
                        y=center_world[1],  # SimpleITK y
                        z=center_world[0],  # SimpleITK z
                        diameter_mm=cand["diameter_mm"],
                        confidence=float(prob),
                        malignancy_score=malignancy,
                    ))

            # Step 5: Non-maximum suppression
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
