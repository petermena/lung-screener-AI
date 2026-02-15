"""DICOM SR feedback loop for radiologist confirmations.

Receives radiologist-verified DICOM Structured Reports back from PACS
to create a feedback loop: confirmed/rejected findings are recorded
and can be used to retrain and improve the model over time.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """A single feedback record from a radiologist."""

    finding_id: str
    series_uid: str
    patient_id: str
    study_date: str
    # Original AI detection values
    ai_x: float
    ai_y: float
    ai_z: float
    ai_diameter_mm: float
    ai_confidence: float
    ai_lung_rads: str
    # Radiologist feedback
    confirmed: bool  # True = true positive, False = false positive
    radiologist_diameter_mm: float = 0.0
    radiologist_lung_rads: str = ""
    radiologist_notes: str = ""
    feedback_date: str = ""
    radiologist_name: str = ""

    def __post_init__(self):
        if not self.feedback_date:
            self.feedback_date = datetime.now().strftime("%Y%m%d")

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "series_uid": self.series_uid,
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "ai_x": self.ai_x,
            "ai_y": self.ai_y,
            "ai_z": self.ai_z,
            "ai_diameter_mm": self.ai_diameter_mm,
            "ai_confidence": self.ai_confidence,
            "ai_lung_rads": self.ai_lung_rads,
            "confirmed": self.confirmed,
            "radiologist_diameter_mm": self.radiologist_diameter_mm,
            "radiologist_lung_rads": self.radiologist_lung_rads,
            "radiologist_notes": self.radiologist_notes,
            "feedback_date": self.feedback_date,
            "radiologist_name": self.radiologist_name,
        }


class FeedbackStore:
    """Persistent storage for radiologist feedback records.

    Stores feedback in a JSON-lines file for easy appending and
    provides methods for querying and aggregating feedback data.
    """

    def __init__(self, feedback_dir: str | Path):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.jsonl"

    def add_record(self, record: FeedbackRecord):
        """Append a feedback record."""
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(
            f"Recorded feedback for {record.series_uid}: "
            f"{'confirmed' if record.confirmed else 'rejected'}"
        )

    def load_all(self) -> list[FeedbackRecord]:
        """Load all feedback records."""
        records = []
        if not self.feedback_file.exists():
            return records

        with open(self.feedback_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(FeedbackRecord(**data))

        return records

    def get_stats(self) -> dict:
        """Compute aggregate feedback statistics."""
        records = self.load_all()
        if not records:
            return {
                "total_feedback": 0,
                "confirmed": 0,
                "rejected": 0,
                "precision": 0.0,
            }

        confirmed = sum(1 for r in records if r.confirmed)
        rejected = sum(1 for r in records if not r.confirmed)

        # Precision = confirmed / total
        precision = confirmed / len(records) if records else 0.0

        # Size agreement
        size_diffs = []
        for r in records:
            if r.confirmed and r.radiologist_diameter_mm > 0:
                size_diffs.append(
                    abs(r.ai_diameter_mm - r.radiologist_diameter_mm)
                )

        avg_size_diff = sum(size_diffs) / len(size_diffs) if size_diffs else 0.0

        # Lung-RADS agreement
        rads_agree = sum(
            1 for r in records
            if r.confirmed and r.radiologist_lung_rads == r.ai_lung_rads
        )
        rads_total = sum(
            1 for r in records
            if r.confirmed and r.radiologist_lung_rads
        )
        rads_agreement = rads_agree / rads_total if rads_total > 0 else 0.0

        # Confidence distribution for TP vs FP
        tp_confidences = [r.ai_confidence for r in records if r.confirmed]
        fp_confidences = [r.ai_confidence for r in records if not r.confirmed]

        return {
            "total_feedback": len(records),
            "confirmed": confirmed,
            "rejected": rejected,
            "precision": round(precision, 4),
            "avg_size_disagreement_mm": round(avg_size_diff, 1),
            "lung_rads_agreement": round(rads_agreement, 4),
            "avg_tp_confidence": (
                round(sum(tp_confidences) / len(tp_confidences), 3)
                if tp_confidences else 0.0
            ),
            "avg_fp_confidence": (
                round(sum(fp_confidences) / len(fp_confidences), 3)
                if fp_confidences else 0.0
            ),
        }

    def export_for_training(self, output_dir: str | Path) -> dict:
        """Export confirmed findings as training annotations.

        Converts confirmed feedback into annotations.csv format that
        can be used to retrain the model with radiologist-verified data.

        Args:
            output_dir: Directory to write exported annotations.

        Returns:
            Dict with export statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = self.load_all()

        # Write confirmed findings as positive annotations
        annotations_path = output_dir / "feedback_annotations.csv"
        with open(annotations_path, "w") as f:
            f.write("seriesuid,coordX,coordY,coordZ,diameter_mm\n")
            confirmed_count = 0
            for r in records:
                if r.confirmed:
                    # Use radiologist diameter if available, else AI diameter
                    diameter = r.radiologist_diameter_mm or r.ai_diameter_mm
                    f.write(
                        f"{r.series_uid},{r.ai_x:.1f},{r.ai_y:.1f},"
                        f"{r.ai_z:.1f},{diameter:.1f}\n"
                    )
                    confirmed_count += 1

        # Write rejected findings as negative candidates
        negatives_path = output_dir / "feedback_negatives.csv"
        with open(negatives_path, "w") as f:
            f.write("seriesuid,coordX,coordY,coordZ,class\n")
            rejected_count = 0
            for r in records:
                if not r.confirmed:
                    f.write(
                        f"{r.series_uid},{r.ai_x:.1f},{r.ai_y:.1f},"
                        f"{r.ai_z:.1f},0\n"
                    )
                    rejected_count += 1

        logger.info(
            f"Exported {confirmed_count} confirmed and {rejected_count} rejected "
            f"findings to {output_dir}"
        )

        return {
            "annotations_file": str(annotations_path),
            "negatives_file": str(negatives_path),
            "confirmed_count": confirmed_count,
            "rejected_count": rejected_count,
        }


def parse_feedback_sr(ds: Dataset) -> list[FeedbackRecord] | None:
    """Parse a DICOM Structured Report for radiologist feedback.

    Looks for radiologist-verified SR documents that contain acceptance
    or rejection of AI-detected findings.

    The expected SR structure:
    - Concept Name: "Imaging Measurement Report" (126000)
    - Contains: Finding items with verification status

    Args:
        ds: pydicom Dataset of the SR.

    Returns:
        List of FeedbackRecord if this is a feedback SR, None otherwise.
    """
    # Check if this is an SR document
    modality = getattr(ds, "Modality", "")
    if modality != "SR":
        return None

    # Check verification flag
    verification = getattr(ds, "VerificationFlag", "")
    if verification != "VERIFIED":
        return None

    content = getattr(ds, "ContentSequence", [])
    if not content:
        return None

    patient_id = str(getattr(ds, "PatientID", ""))
    study_date = str(getattr(ds, "StudyDate", ""))
    radiologist_name = ""

    # Look for verifying observer
    for item in content:
        concept_name = _get_concept_name(item)
        if concept_name == "Verifying Observer Name":
            radiologist_name = str(getattr(item, "TextValue", ""))
            break

    records = []
    finding_num = 0

    for item in content:
        if getattr(item, "ValueType", "") != "CONTAINER":
            continue

        concept_name = _get_concept_name(item)
        if concept_name != "Finding":
            continue

        finding_num += 1
        sub_content = getattr(item, "ContentSequence", [])
        if not sub_content:
            continue

        # Extract finding details
        series_uid = ""
        x = y = z = 0.0
        diameter_mm = 0.0
        confidence = 0.0
        lung_rads = ""
        confirmed = True  # Default to confirmed if verified SR
        rad_diameter = 0.0
        rad_rads = ""
        notes = ""

        for sub_item in sub_content:
            sub_concept = _get_concept_name(sub_item)

            if sub_concept == "Path Length":
                measured = getattr(sub_item, "MeasuredValueSequence", [])
                if measured:
                    diameter_mm = float(getattr(measured[0], "NumericValue", 0))

            elif sub_concept == "Detection Confidence":
                measured = getattr(sub_item, "MeasuredValueSequence", [])
                if measured:
                    confidence = float(getattr(measured[0], "NumericValue", 0))

            elif sub_concept == "Lung-RADS Category":
                lung_rads = str(getattr(sub_item, "TextValue", ""))

            elif sub_concept == "Location of Measurement":
                loc_text = str(getattr(sub_item, "TextValue", ""))
                coords = _parse_coordinates(loc_text)
                if coords:
                    x, y, z = coords

            elif sub_concept == "Verification Status":
                status = str(getattr(sub_item, "TextValue", "")).lower()
                confirmed = status in ("confirmed", "true positive", "accepted")

            elif sub_concept == "Radiologist Measurement":
                measured = getattr(sub_item, "MeasuredValueSequence", [])
                if measured:
                    rad_diameter = float(getattr(measured[0], "NumericValue", 0))

            elif sub_concept == "Radiologist Lung-RADS":
                rad_rads = str(getattr(sub_item, "TextValue", ""))

            elif sub_concept == "Comment":
                notes = str(getattr(sub_item, "TextValue", ""))

        record = FeedbackRecord(
            finding_id=f"{series_uid}_{finding_num}",
            series_uid=series_uid,
            patient_id=patient_id,
            study_date=study_date,
            ai_x=x,
            ai_y=y,
            ai_z=z,
            ai_diameter_mm=diameter_mm,
            ai_confidence=confidence,
            ai_lung_rads=lung_rads,
            confirmed=confirmed,
            radiologist_diameter_mm=rad_diameter,
            radiologist_lung_rads=rad_rads,
            radiologist_notes=notes,
            radiologist_name=radiologist_name,
        )
        records.append(record)

    return records if records else None


def _get_concept_name(item: Dataset) -> str:
    """Extract the CodeMeaning from a content item's ConceptNameCodeSequence."""
    seq = getattr(item, "ConceptNameCodeSequence", [])
    if seq:
        return str(getattr(seq[0], "CodeMeaning", ""))
    return ""


def _parse_coordinates(text: str) -> tuple[float, float, float] | None:
    """Parse coordinates from text like '(1.0, 2.0, 3.0) mm'."""
    try:
        # Remove parentheses and units
        text = text.strip()
        if text.startswith("("):
            text = text.split(")")[0].lstrip("(")
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (ValueError, IndexError):
        pass
    return None
