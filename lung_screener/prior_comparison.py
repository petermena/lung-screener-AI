"""Prior study comparison and nodule growth tracking.

Enables longitudinal tracking of nodules across serial CT exams to
detect growth, compute volume doubling time (VDT), and flag concerning
interval changes.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NoduleMeasurement:
    """A single measurement of a nodule at one time point."""

    study_date: str  # YYYYMMDD
    diameter_mm: float
    x: float
    y: float
    z: float
    confidence: float = 0.0
    nodule_type: str = "solid"
    lung_rads: str = ""
    series_uid: str = ""
    volume_mm3: float = 0.0

    def __post_init__(self):
        if self.volume_mm3 == 0.0 and self.diameter_mm > 0:
            # Estimate volume assuming sphere
            r = self.diameter_mm / 2.0
            self.volume_mm3 = (4.0 / 3.0) * math.pi * r ** 3

    def to_dict(self) -> dict:
        return {
            "study_date": self.study_date,
            "diameter_mm": round(self.diameter_mm, 1),
            "volume_mm3": round(self.volume_mm3, 1),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "z": round(self.z, 1),
            "confidence": round(self.confidence, 3),
            "nodule_type": self.nodule_type,
            "lung_rads": self.lung_rads,
            "series_uid": self.series_uid,
        }


@dataclass
class NoduleTrack:
    """A tracked nodule across multiple time points."""

    track_id: str
    patient_id: str
    lobe: str = ""
    measurements: list[NoduleMeasurement] = field(default_factory=list)

    @property
    def latest(self) -> NoduleMeasurement | None:
        if not self.measurements:
            return None
        return max(self.measurements, key=lambda m: m.study_date)

    @property
    def earliest(self) -> NoduleMeasurement | None:
        if not self.measurements:
            return None
        return min(self.measurements, key=lambda m: m.study_date)

    def volume_doubling_time_days(self) -> float | None:
        """Compute volume doubling time (VDT) between earliest and latest.

        VDT = (delta_t * ln(2)) / ln(V2 / V1)

        Returns:
            VDT in days, or None if insufficient data.
            Negative VDT indicates shrinkage.
        """
        if len(self.measurements) < 2:
            return None

        earliest = self.earliest
        latest = self.latest

        if earliest.volume_mm3 <= 0 or latest.volume_mm3 <= 0:
            return None

        # Parse dates
        try:
            d1 = datetime.strptime(earliest.study_date, "%Y%m%d")
            d2 = datetime.strptime(latest.study_date, "%Y%m%d")
        except ValueError:
            return None

        delta_days = (d2 - d1).days
        if delta_days <= 0:
            return None

        volume_ratio = latest.volume_mm3 / earliest.volume_mm3
        if volume_ratio <= 0:
            return None
        if abs(volume_ratio - 1.0) < 0.001:
            return float("inf")  # No change

        vdt = (delta_days * math.log(2)) / math.log(volume_ratio)
        return vdt

    def diameter_change_mm(self) -> float | None:
        """Compute absolute diameter change from earliest to latest."""
        if len(self.measurements) < 2:
            return None
        return self.latest.diameter_mm - self.earliest.diameter_mm

    def diameter_change_percent(self) -> float | None:
        """Compute percent diameter change."""
        if len(self.measurements) < 2:
            return None
        if self.earliest.diameter_mm <= 0:
            return None
        return (
            (self.latest.diameter_mm - self.earliest.diameter_mm)
            / self.earliest.diameter_mm
            * 100.0
        )

    def growth_assessment(self) -> str:
        """Classify growth pattern for clinical reporting.

        Returns one of:
        - "new": Only one measurement
        - "stable": Diameter change < 1.5mm and VDT > 600 days
        - "slow_growth": VDT 400-600 days
        - "growing": VDT < 400 days or diameter increase >= 2mm
        - "shrinking": Diameter decrease > 1mm
        """
        if len(self.measurements) < 2:
            return "new"

        diameter_change = self.diameter_change_mm()
        vdt = self.volume_doubling_time_days()

        if diameter_change is not None and diameter_change < -1.0:
            return "shrinking"

        if vdt is not None and vdt < 0:
            return "shrinking"

        if diameter_change is not None and abs(diameter_change) < 1.5:
            if vdt is None or vdt > 600:
                return "stable"

        if vdt is not None:
            if vdt < 400:
                return "growing"
            elif vdt < 600:
                return "slow_growth"

        if diameter_change is not None and diameter_change >= 2.0:
            return "growing"

        return "stable"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "patient_id": self.patient_id,
            "lobe": self.lobe,
            "num_timepoints": len(self.measurements),
            "measurements": [m.to_dict() for m in sorted(
                self.measurements, key=lambda m: m.study_date
            )],
            "growth_assessment": self.growth_assessment(),
            "volume_doubling_time_days": self.volume_doubling_time_days(),
            "diameter_change_mm": self.diameter_change_mm(),
            "diameter_change_percent": self.diameter_change_percent(),
        }


class PriorStudyTracker:
    """Manages longitudinal nodule tracking across studies.

    Stores tracked nodules in a JSON file per patient, matching new
    findings to previously identified nodules by spatial proximity.
    """

    MATCH_DISTANCE_MM = 15.0  # Max distance to consider same nodule

    def __init__(self, tracking_dir: str | Path):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def _patient_file(self, patient_id: str) -> Path:
        safe_id = patient_id.replace("/", "_").replace("\\", "_")
        return self.tracking_dir / f"{safe_id}.json"

    def load_patient_tracks(self, patient_id: str) -> list[NoduleTrack]:
        """Load all tracked nodules for a patient."""
        path = self._patient_file(patient_id)
        if not path.exists():
            return []

        with open(path) as f:
            data = json.load(f)

        tracks = []
        for track_data in data.get("tracks", []):
            measurements = [
                NoduleMeasurement(**m) for m in track_data.get("measurements", [])
            ]
            tracks.append(NoduleTrack(
                track_id=track_data["track_id"],
                patient_id=patient_id,
                lobe=track_data.get("lobe", ""),
                measurements=measurements,
            ))

        return tracks

    def save_patient_tracks(self, patient_id: str, tracks: list[NoduleTrack]):
        """Save all tracked nodules for a patient."""
        path = self._patient_file(patient_id)
        data = {
            "patient_id": patient_id,
            "last_updated": datetime.now().isoformat(),
            "tracks": [t.to_dict() for t in tracks],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def match_and_update(
        self,
        patient_id: str,
        study_date: str,
        findings: list,
        series_uid: str = "",
    ) -> list[NoduleTrack]:
        """Match new findings to existing tracks and update.

        For each finding in the new study:
        1. Find the closest existing track within MATCH_DISTANCE_MM
        2. If matched, add the new measurement to that track
        3. If unmatched, create a new track

        Args:
            patient_id: Patient identifier.
            study_date: Study date in YYYYMMDD format.
            findings: List of NoduleFinding objects from inference.
            series_uid: Series UID of the current study.

        Returns:
            Updated list of NoduleTrack objects.
        """
        tracks = self.load_patient_tracks(patient_id)
        matched_track_ids = set()

        for finding in findings:
            measurement = NoduleMeasurement(
                study_date=study_date,
                diameter_mm=finding.diameter_mm,
                x=finding.x,
                y=finding.y,
                z=finding.z,
                confidence=finding.confidence,
                nodule_type=getattr(finding, "nodule_type", "solid"),
                lung_rads=finding.lung_rads,
                series_uid=series_uid,
            )

            # Find closest matching track
            best_track = None
            best_distance = float("inf")

            for track in tracks:
                if track.track_id in matched_track_ids:
                    continue
                latest = track.latest
                if latest is None:
                    continue

                dist = math.sqrt(
                    (finding.x - latest.x) ** 2
                    + (finding.y - latest.y) ** 2
                    + (finding.z - latest.z) ** 2
                )
                if dist < best_distance and dist <= self.MATCH_DISTANCE_MM:
                    best_distance = dist
                    best_track = track

            if best_track is not None:
                # Check that this study_date isn't already recorded
                existing_dates = {m.study_date for m in best_track.measurements}
                if study_date not in existing_dates:
                    best_track.measurements.append(measurement)
                matched_track_ids.add(best_track.track_id)
                logger.info(
                    f"Matched finding to track {best_track.track_id} "
                    f"(distance: {best_distance:.1f} mm)"
                )
            else:
                # Create new track
                import uuid
                track_id = str(uuid.uuid4())[:8]
                new_track = NoduleTrack(
                    track_id=track_id,
                    patient_id=patient_id,
                    lobe=getattr(finding, "lobe", ""),
                    measurements=[measurement],
                )
                tracks.append(new_track)
                logger.info(f"Created new track {track_id} for unmatched finding")

        # Save updated tracks
        self.save_patient_tracks(patient_id, tracks)
        return tracks


def format_comparison_report(tracks: list[NoduleTrack]) -> str:
    """Generate a comparison report section for radiology dictation.

    Args:
        tracks: List of NoduleTrack objects with current and prior data.

    Returns:
        Report text describing interval changes.
    """
    if not tracks:
        return "No prior studies available for comparison."

    lines = ["COMPARISON WITH PRIOR STUDIES:", ""]

    has_priors = any(len(t.measurements) > 1 for t in tracks)
    if not has_priors:
        lines.append("No prior CT studies available for comparison.")
        return "\n".join(lines)

    for track in tracks:
        if len(track.measurements) < 2:
            latest = track.latest
            if latest:
                lines.append(
                    f"- {track.lobe.capitalize() or 'Unknown lobe'}: "
                    f"New {latest.diameter_mm:.0f} mm nodule (no prior for comparison)."
                )
            continue

        latest = track.latest
        earliest = track.earliest
        growth = track.growth_assessment()
        vdt = track.volume_doubling_time_days()
        d_change = track.diameter_change_mm()

        lobe = track.lobe.capitalize() or "Unknown lobe"
        desc = f"- {lobe}: {latest.diameter_mm:.0f} mm"

        if growth == "stable":
            desc += (
                f", previously {earliest.diameter_mm:.0f} mm "
                f"on {_format_date(earliest.study_date)}. Stable."
            )
        elif growth == "growing":
            desc += (
                f", previously {earliest.diameter_mm:.0f} mm "
                f"on {_format_date(earliest.study_date)}. "
                f"INTERVAL GROWTH ({d_change:+.1f} mm)."
            )
            if vdt is not None and vdt > 0:
                desc += f" Volume doubling time: {vdt:.0f} days."
        elif growth == "slow_growth":
            desc += (
                f", previously {earliest.diameter_mm:.0f} mm "
                f"on {_format_date(earliest.study_date)}. "
                f"Slow interval growth ({d_change:+.1f} mm)."
            )
        elif growth == "shrinking":
            desc += (
                f", previously {earliest.diameter_mm:.0f} mm "
                f"on {_format_date(earliest.study_date)}. Decreased in size."
            )

        lines.append(desc)

    return "\n".join(lines)


def _format_date(yyyymmdd: str) -> str:
    """Format YYYYMMDD as readable date."""
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        return yyyymmdd
