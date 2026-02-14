"""Data management for manual DICOM upload and annotation.

Provides a workflow for building a local training dataset from DICOM scans:

1. Import: Copy a DICOM series directory, convert to .mhd format,
   and register it in the local dataset.
2. Annotate: Record nodule locations and sizes for each scan.
3. Prepare: Compile annotations into the CSV files the trainer expects
   (annotations.csv + candidates_V2.csv).

All state is stored in data/training/manifest.json so you can resume
across sessions.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from .preprocessing import CTPreprocessor, extract_candidates, load_dicom_series

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


class DataManager:
    """Manages a local training dataset built from imported DICOM scans."""

    def __init__(self, data_dir: str | Path, config: dict):
        self.data_dir = Path(data_dir)
        self.scans_dir = self.data_dir / "scans"
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.preprocessor = CTPreprocessor(config)
        self.manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest persistence
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.data_dir / MANIFEST_FILENAME

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"scans": {}, "created": datetime.now().isoformat()}

    def _save_manifest(self):
        with open(self._manifest_path(), "w") as f:
            json.dump(self.manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_dicom(self, dicom_dir: str | Path, label: str = "") -> dict:
        """Import a DICOM series into the training dataset.

        Reads the DICOM directory, extracts patient/study metadata,
        converts to .mhd format for training, and registers the scan.

        Args:
            dicom_dir: Path to a directory of DICOM files (one series).
            label: Optional human-readable label (e.g. "patient_042").

        Returns:
            Dict with scan metadata.
        """
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {dicom_dir}")

        # Read DICOM metadata from first file
        dcm_files = sorted(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            # Try without .dcm extension (some DICOM files have no extension)
            dcm_files = [
                f for f in sorted(dicom_dir.iterdir())
                if f.is_file() and not f.name.startswith(".")
            ]
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        series_uid = str(getattr(ds, "SeriesInstanceUID", dicom_dir.name))
        patient_id = str(getattr(ds, "PatientID", "unknown"))
        study_date = str(getattr(ds, "StudyDate", ""))
        modality = str(getattr(ds, "Modality", "CT"))

        if series_uid in self.manifest["scans"]:
            logger.warning(f"Series {series_uid} already imported, skipping")
            return self.manifest["scans"][series_uid]

        # Load as SimpleITK image
        logger.info(f"Loading DICOM series from {dicom_dir} ({len(dcm_files)} files)...")
        image = load_dicom_series(dicom_dir)

        # Save as .mhd/.raw in the scans directory
        scan_dir = self.scans_dir / series_uid
        scan_dir.mkdir(parents=True, exist_ok=True)
        mhd_path = scan_dir / f"{series_uid}.mhd"

        logger.info(f"Converting to MHD format: {mhd_path}")
        sitk.WriteImage(image, str(mhd_path))

        # Also keep a copy of the original DICOMs for annotation reference
        dicom_copy_dir = scan_dir / "dicom"
        if not dicom_copy_dir.exists():
            logger.info("Copying original DICOM files...")
            shutil.copytree(dicom_dir, dicom_copy_dir)

        # Extract spatial info
        spacing = list(image.GetSpacing())
        origin = list(image.GetOrigin())
        size = list(image.GetSize())

        scan_meta = {
            "series_uid": series_uid,
            "patient_id": patient_id,
            "study_date": study_date,
            "modality": modality,
            "label": label or patient_id,
            "num_slices": len(dcm_files),
            "size": size,
            "spacing": spacing,
            "origin": origin,
            "mhd_path": str(mhd_path.relative_to(self.data_dir)),
            "dicom_dir": str(dicom_copy_dir.relative_to(self.data_dir)),
            "imported_at": datetime.now().isoformat(),
            "annotations": [],
            "status": "imported",
        }

        self.manifest["scans"][series_uid] = scan_meta
        self._save_manifest()

        logger.info(
            f"Imported: {series_uid} | Patient: {patient_id} | "
            f"Slices: {len(dcm_files)} | Size: {size}"
        )
        return scan_meta

    # ------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------

    def annotate(
        self,
        series_uid: str,
        x: float,
        y: float,
        z: float,
        diameter_mm: float,
        note: str = "",
    ) -> dict:
        """Add a nodule annotation to an imported scan.

        Coordinates are world coordinates in mm (the same system used by
        DICOM ImagePositionPatient). You can get these from any DICOM viewer
        that shows cursor position in mm.

        Args:
            series_uid: Series UID of the imported scan.
            x: X coordinate in mm (left-right).
            y: Y coordinate in mm (anterior-posterior).
            z: Z coordinate in mm (head-foot / slice position).
            diameter_mm: Estimated nodule diameter in mm.
            note: Optional clinical note.

        Returns:
            The annotation dict.
        """
        if series_uid not in self.manifest["scans"]:
            raise ValueError(f"Unknown series: {series_uid}. Import it first.")

        annotation = {
            "coordX": round(x, 2),
            "coordY": round(y, 2),
            "coordZ": round(z, 2),
            "diameter_mm": round(diameter_mm, 1),
            "note": note,
            "added_at": datetime.now().isoformat(),
        }

        self.manifest["scans"][series_uid]["annotations"].append(annotation)
        self.manifest["scans"][series_uid]["status"] = "annotated"
        self._save_manifest()

        logger.info(
            f"Added annotation to {series_uid}: "
            f"({x:.1f}, {y:.1f}, {z:.1f}) mm, {diameter_mm:.1f}mm diameter"
        )
        return annotation

    def mark_negative(self, series_uid: str):
        """Mark a scan as having no nodules (negative case).

        This is important for training — the model needs negative examples.
        """
        if series_uid not in self.manifest["scans"]:
            raise ValueError(f"Unknown series: {series_uid}. Import it first.")

        self.manifest["scans"][series_uid]["annotations"] = []
        self.manifest["scans"][series_uid]["status"] = "negative"
        self._save_manifest()
        logger.info(f"Marked {series_uid} as negative (no nodules)")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_scans(self) -> list[dict]:
        """Return a summary of all imported scans."""
        summaries = []
        for uid, scan in self.manifest["scans"].items():
            summaries.append({
                "series_uid": uid,
                "label": scan.get("label", ""),
                "patient_id": scan.get("patient_id", ""),
                "study_date": scan.get("study_date", ""),
                "num_slices": scan.get("num_slices", 0),
                "num_annotations": len(scan.get("annotations", [])),
                "status": scan.get("status", "unknown"),
            })
        return summaries

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------

    def prepare(self, output_dir: str | Path | None = None) -> Path:
        """Compile imported scans and annotations into training-ready format.

        Generates:
          - subset0/ directory with symlinks to .mhd files
          - annotations.csv with positive nodule locations
          - candidates_V2.csv with both positive and auto-detected negative
            candidate locations

        This output directory can be used directly as dataset_dir for training.

        Args:
            output_dir: Where to write training data. Defaults to data_dir/prepared.

        Returns:
            Path to the prepared dataset directory.
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir / "prepared"
        subset_dir = output_dir / "subset0"
        subset_dir.mkdir(parents=True, exist_ok=True)

        annotations_rows = []
        candidates_rows = []

        scans = self.manifest["scans"]
        annotated = [
            s for s in scans.values() if s["status"] in ("annotated", "negative")
        ]

        if not annotated:
            raise RuntimeError(
                "No annotated scans found. Import scans and add annotations first."
            )

        logger.info(f"Preparing training data from {len(annotated)} scans...")

        for scan in annotated:
            series_uid = scan["series_uid"]
            mhd_src = self.data_dir / scan["mhd_path"]

            if not mhd_src.exists():
                logger.warning(f"MHD file missing for {series_uid}, skipping")
                continue

            # Symlink .mhd and .raw into subset0/
            mhd_dst = subset_dir / mhd_src.name
            raw_src = mhd_src.with_suffix(".raw")
            raw_dst = subset_dir / raw_src.name

            for src, dst in [(mhd_src, mhd_dst), (raw_src, raw_dst)]:
                if dst.exists():
                    dst.unlink()
                if src.exists():
                    dst.symlink_to(src.resolve())

            # Build annotation rows from manual annotations
            for ann in scan.get("annotations", []):
                annotations_rows.append({
                    "seriesuid": series_uid,
                    "coordX": ann["coordX"],
                    "coordY": ann["coordY"],
                    "coordZ": ann["coordZ"],
                    "diameter_mm": ann["diameter_mm"],
                })

                # Also add as positive candidate
                candidates_rows.append({
                    "seriesuid": series_uid,
                    "coordX": ann["coordX"],
                    "coordY": ann["coordY"],
                    "coordZ": ann["coordZ"],
                    "class": 1,
                })

            # Run candidate detection to get negative candidates
            logger.info(f"Extracting negative candidates from {series_uid}...")
            image = sitk.ReadImage(str(mhd_src))
            processed = self.preprocessor.process_scan(image)

            positive_coords = np.array([
                [a["coordX"], a["coordY"], a["coordZ"]]
                for a in scan.get("annotations", [])
            ]) if scan.get("annotations") else np.empty((0, 3))

            for cand in processed["candidates"]:
                cw = cand["center_world"]
                # world coords from preprocessor are (z, y, x) — convert to (x, y, z)
                cx, cy, cz = cw[2], cw[1], cw[0]

                # Check if this candidate matches a positive annotation
                is_positive = False
                if len(positive_coords) > 0:
                    dists = np.sqrt(np.sum(
                        (positive_coords - np.array([cx, cy, cz])) ** 2, axis=1
                    ))
                    if np.min(dists) < 5.0:
                        is_positive = True

                if not is_positive:
                    candidates_rows.append({
                        "seriesuid": series_uid,
                        "coordX": round(cx, 2),
                        "coordY": round(cy, 2),
                        "coordZ": round(cz, 2),
                        "class": 0,
                    })

        # Write CSVs
        import csv

        ann_path = output_dir / "annotations.csv"
        with open(ann_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
            writer.writeheader()
            writer.writerows(annotations_rows)

        cand_path = output_dir / "candidates_V2.csv"
        with open(cand_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seriesuid", "coordX", "coordY", "coordZ", "class"])
            writer.writeheader()
            writer.writerows(candidates_rows)

        num_pos = sum(1 for c in candidates_rows if c["class"] == 1)
        num_neg = sum(1 for c in candidates_rows if c["class"] == 0)

        logger.info(f"Training data prepared in {output_dir}")
        logger.info(f"  Scans: {len(annotated)}")
        logger.info(f"  Annotations (nodules): {len(annotations_rows)}")
        logger.info(f"  Candidates: {num_pos} positive, {num_neg} negative")
        logger.info(f"  annotations.csv: {ann_path}")
        logger.info(f"  candidates_V2.csv: {cand_path}")

        return output_dir
