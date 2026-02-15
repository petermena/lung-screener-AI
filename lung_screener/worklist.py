"""Worklist and batch processing for multiple CT studies.

Provides a prioritized worklist for processing multiple studies,
with progress tracking, result aggregation, and priority scheduling
based on clinical urgency.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import SimpleITK as sitk

from .inference import NoduleDetector, ScanResult
from .input_validation import validate_dicom_series
from .preprocessing import load_dicom_series, load_mhd

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Study processing priority levels."""
    STAT = 0
    URGENT = 1
    ROUTINE = 2
    LOW = 3


class StudyStatus(Enum):
    """Processing status for a study."""
    PENDING = "pending"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorklistEntry:
    """A single study in the processing worklist."""

    study_id: str
    input_path: str
    priority: Priority = Priority.ROUTINE
    status: StudyStatus = StudyStatus.PENDING
    series_uid: str = ""
    patient_id: str = ""
    result: ScanResult | None = None
    error_message: str = ""
    processing_time_s: float = 0.0
    queued_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.queued_at:
            self.queued_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = {
            "study_id": self.study_id,
            "input_path": self.input_path,
            "priority": self.priority.name,
            "status": self.status.value,
            "series_uid": self.series_uid,
            "patient_id": self.patient_id,
            "error_message": self.error_message,
            "processing_time_s": round(self.processing_time_s, 2),
            "queued_at": self.queued_at,
            "completed_at": self.completed_at,
        }
        if self.result:
            d["result"] = self.result.to_dict()
        return d


class BatchProcessor:
    """Batch processing engine for multiple CT studies.

    Processes studies from a worklist in priority order, with
    input validation, error handling, and result aggregation.
    """

    def __init__(
        self,
        detector: NoduleDetector,
        validate_input: bool = True,
        on_result: callable | None = None,
        on_progress: callable | None = None,
    ):
        self.detector = detector
        self.validate_input = validate_input
        self.on_result = on_result
        self.on_progress = on_progress
        self.worklist: list[WorklistEntry] = []

    def add_study(
        self,
        input_path: str | Path,
        priority: Priority = Priority.ROUTINE,
        study_id: str = "",
        patient_id: str = "",
    ) -> WorklistEntry:
        """Add a study to the worklist.

        Args:
            input_path: Path to DICOM directory or .mhd file.
            priority: Processing priority.
            study_id: Optional study identifier.
            patient_id: Optional patient identifier.

        Returns:
            WorklistEntry added to the queue.
        """
        input_path = Path(input_path)
        if not study_id:
            study_id = input_path.name

        entry = WorklistEntry(
            study_id=study_id,
            input_path=str(input_path),
            priority=priority,
            patient_id=patient_id,
        )
        self.worklist.append(entry)
        logger.info(f"Added study {study_id} to worklist (priority: {priority.name})")
        return entry

    def add_directory(
        self,
        root_dir: str | Path,
        priority: Priority = Priority.ROUTINE,
    ) -> int:
        """Add all DICOM study directories under root_dir.

        Scans for directories containing .dcm files and adds each as a study.

        Args:
            root_dir: Root directory to scan.
            priority: Priority for all discovered studies.

        Returns:
            Number of studies added.
        """
        root_dir = Path(root_dir)
        count = 0

        for path in sorted(root_dir.iterdir()):
            if path.is_dir():
                dcm_files = list(path.glob("*.dcm"))
                if dcm_files:
                    self.add_study(path, priority=priority)
                    count += 1
                else:
                    # Check subdirectories (study/series structure)
                    for subdir in path.iterdir():
                        if subdir.is_dir() and list(subdir.glob("*.dcm")):
                            self.add_study(subdir, priority=priority)
                            count += 1
            elif path.suffix == ".mhd":
                self.add_study(path, priority=priority)
                count += 1

        logger.info(f"Added {count} studies from {root_dir}")
        return count

    def process_all(self) -> list[WorklistEntry]:
        """Process all pending studies in priority order.

        Returns:
            List of all WorklistEntry objects with results.
        """
        # Sort by priority (lower enum value = higher priority)
        pending = [e for e in self.worklist if e.status == StudyStatus.PENDING]
        pending.sort(key=lambda e: e.priority.value)

        total = len(pending)
        logger.info(f"Processing {total} studies")

        for idx, entry in enumerate(pending):
            self._notify_progress(idx, total, entry)
            self._process_entry(entry)
            self._notify_progress(idx + 1, total, entry)

        return self.worklist

    def _process_entry(self, entry: WorklistEntry):
        """Process a single worklist entry."""
        start_time = time.time()

        try:
            input_path = Path(entry.input_path)

            # Validate input
            if self.validate_input and input_path.is_dir():
                entry.status = StudyStatus.VALIDATING
                validation = validate_dicom_series(input_path)
                if not validation.is_valid:
                    entry.status = StudyStatus.SKIPPED
                    entry.error_message = "; ".join(validation.errors)
                    logger.warning(
                        f"Skipping {entry.study_id}: {entry.error_message}"
                    )
                    return

            # Load scan
            entry.status = StudyStatus.PROCESSING
            if input_path.suffix == ".mhd":
                image = load_mhd(input_path)
                entry.series_uid = input_path.stem
            elif input_path.is_dir():
                image = load_dicom_series(input_path)
                entry.series_uid = input_path.name
            else:
                entry.status = StudyStatus.FAILED
                entry.error_message = f"Unsupported input: {input_path}"
                return

            # Run detection
            result = self.detector.predict_scan(
                image, series_uid=entry.series_uid
            )
            entry.result = result
            entry.status = StudyStatus.COMPLETED
            entry.completed_at = datetime.now().isoformat()
            entry.processing_time_s = time.time() - start_time

            logger.info(
                f"Completed {entry.study_id}: {len(result.findings)} findings, "
                f"Lung-RADS {result.lung_rads_overall or '1'} "
                f"({entry.processing_time_s:.1f}s)"
            )

            if self.on_result:
                self.on_result(entry)

        except Exception as e:
            entry.status = StudyStatus.FAILED
            entry.error_message = str(e)
            entry.processing_time_s = time.time() - start_time
            logger.error(f"Failed {entry.study_id}: {e}")

    def _notify_progress(self, current: int, total: int, entry: WorklistEntry):
        if self.on_progress:
            self.on_progress(current, total, entry)

    def summary(self) -> dict:
        """Generate a summary of all processed studies."""
        completed = [e for e in self.worklist if e.status == StudyStatus.COMPLETED]
        failed = [e for e in self.worklist if e.status == StudyStatus.FAILED]
        skipped = [e for e in self.worklist if e.status == StudyStatus.SKIPPED]
        pending = [e for e in self.worklist if e.status == StudyStatus.PENDING]

        # Aggregate findings
        total_findings = sum(
            len(e.result.findings) for e in completed if e.result
        )
        lung_rads_dist = {}
        for e in completed:
            if e.result:
                cat = e.result.lung_rads_overall or "1"
                lung_rads_dist[cat] = lung_rads_dist.get(cat, 0) + 1

        # Flag high-priority results
        actionable = [
            e for e in completed
            if e.result and e.result.lung_rads_overall in ("3", "4A", "4B")
        ]

        total_time = sum(e.processing_time_s for e in completed)

        return {
            "total_studies": len(self.worklist),
            "completed": len(completed),
            "failed": len(failed),
            "skipped": len(skipped),
            "pending": len(pending),
            "total_findings": total_findings,
            "actionable_studies": len(actionable),
            "lung_rads_distribution": lung_rads_dist,
            "total_processing_time_s": round(total_time, 1),
            "avg_processing_time_s": round(total_time / max(1, len(completed)), 1),
        }

    def save_results(self, output_path: str | Path):
        """Save all results to JSON."""
        data = {
            "summary": self.summary(),
            "studies": [e.to_dict() for e in self.worklist],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def format_worklist_report(self) -> str:
        """Generate a human-readable worklist report."""
        lines = ["BATCH PROCESSING REPORT", "=" * 60, ""]

        summary = self.summary()
        lines.append(f"Total Studies: {summary['total_studies']}")
        lines.append(f"Completed: {summary['completed']}")
        lines.append(f"Failed: {summary['failed']}")
        lines.append(f"Skipped: {summary['skipped']}")
        lines.append(f"Total Findings: {summary['total_findings']}")
        lines.append(f"Actionable (Lung-RADS 3+): {summary['actionable_studies']}")
        lines.append(
            f"Processing Time: {summary['total_processing_time_s']:.1f}s "
            f"(avg {summary['avg_processing_time_s']:.1f}s/study)"
        )
        lines.append("")

        if summary["lung_rads_distribution"]:
            lines.append("Lung-RADS Distribution:")
            for cat in sorted(summary["lung_rads_distribution"]):
                count = summary["lung_rads_distribution"][cat]
                lines.append(f"  Category {cat}: {count}")
            lines.append("")

        # Actionable studies first
        actionable = [
            e for e in self.worklist
            if e.status == StudyStatus.COMPLETED
            and e.result and e.result.lung_rads_overall in ("3", "4A", "4B")
        ]
        if actionable:
            lines.append("ACTIONABLE STUDIES (requires follow-up):")
            lines.append("-" * 40)
            for e in actionable:
                lines.append(
                    f"  {e.study_id}: Lung-RADS {e.result.lung_rads_overall}, "
                    f"{len(e.result.findings)} finding(s)"
                )
            lines.append("")

        # Failed studies
        if summary["failed"] > 0:
            lines.append("FAILED STUDIES:")
            lines.append("-" * 40)
            for e in self.worklist:
                if e.status == StudyStatus.FAILED:
                    lines.append(f"  {e.study_id}: {e.error_message}")
            lines.append("")

        return "\n".join(lines)
