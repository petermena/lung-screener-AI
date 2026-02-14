"""DICOM networking layer for PACS integration.

Provides:
- DICOM Storage SCP: Receives CT studies pushed from PACS
- DICOM SR generation: Creates Structured Reports with findings
- DICOM C-STORE SCU: Sends results back to PACS

Designed for integration with GE Centricity PACS via standard
DICOM networking protocols.
"""

import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import (
    CTImageStorage,
    ComprehensiveSRStorage,
    Verification,
)

from .annotations import create_annotated_slices, create_gsps
from .inference import NoduleDetector, ScanResult
from .preprocessing import load_dicom_series

logger = logging.getLogger(__name__)


class DicomStorageSCP:
    """DICOM Storage SCP that receives CT studies from PACS.

    Listens for incoming C-STORE requests, saves DICOM files to disk,
    and triggers nodule detection when a complete series is received.
    """

    def __init__(
        self,
        config: dict,
        detector: NoduleDetector,
        on_result: callable | None = None,
    ):
        pacs_config = config.get("pacs", {})
        self.ae_title = pacs_config.get("local_ae_title", "LUNG_SCREEN_AI")
        self.port = pacs_config.get("local_port", 11112)
        self.storage_dir = Path(pacs_config.get("storage_dir", "./data/incoming"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.detector = detector
        self.on_result = on_result
        self.config = config

        # Track incoming series
        self._series_files: dict[str, list[Path]] = {}
        self._series_timers: dict[str, float] = {}

        # Set up Application Entity
        self.ae = AE(ae_title=self.ae_title)

        # Accept CT Image Storage
        self.ae.supported_contexts = StoragePresentationContexts

        # Also support verification (C-ECHO)
        self.ae.add_supported_context(Verification)

    def _handle_store(self, event: evt.Event) -> int:
        """Handle incoming C-STORE request."""
        ds = event.dataset
        ds.file_meta = event.file_meta

        series_uid = str(ds.SeriesInstanceUID)
        study_uid = str(ds.StudyInstanceUID)
        sop_uid = str(ds.SOPInstanceUID)

        # Save to disk organized by study/series
        series_dir = self.storage_dir / study_uid / series_uid
        series_dir.mkdir(parents=True, exist_ok=True)
        file_path = series_dir / f"{sop_uid}.dcm"

        ds.save_as(file_path, write_like_original=False)

        # Track files per series
        if series_uid not in self._series_files:
            self._series_files[series_uid] = []
        self._series_files[series_uid].append(file_path)

        logger.debug(
            f"Received instance {sop_uid} for series {series_uid} "
            f"({len(self._series_files[series_uid])} files)"
        )

        return 0x0000  # Success

    def _handle_release(self, event: evt.Event):
        """Handle association release - process any pending series."""
        for series_uid, files in self._series_files.items():
            if len(files) > 10:  # Minimum slices for a meaningful CT
                self._process_series(series_uid)

        # Clear tracking
        self._series_files.clear()

    def _process_series(self, series_uid: str):
        """Process a complete series through the detection pipeline."""
        files = self._series_files.get(series_uid, [])
        if not files:
            return

        series_dir = files[0].parent
        logger.info(f"Processing series {series_uid} ({len(files)} slices)")

        try:
            # Load DICOM series
            image = load_dicom_series(series_dir)

            # Run detection
            result = self.detector.predict_scan(image, series_uid=series_uid)

            logger.info(result.summary())

            # Generate and send DICOM SR
            if result.findings:
                self._send_results(result, series_dir)

            # Callback
            if self.on_result:
                self.on_result(result)

        except Exception as e:
            logger.error(f"Error processing series {series_uid}: {e}")

    def _send_results(self, result: ScanResult, series_dir: Path):
        """Generate DICOM SR, annotated images, and GSPS, then send to PACS."""
        ref_files = list(series_dir.glob("*.dcm"))
        if not ref_files:
            return

        ref_ds = pydicom.dcmread(ref_files[0])

        # Read all DICOM slices for annotation rendering
        all_slices = [pydicom.dcmread(f) for f in ref_files]

        # Create all result objects
        sr = create_nodule_sr(result, ref_ds)
        annotated_sc = create_annotated_slices(result, all_slices)
        gsps_list = create_gsps(result, all_slices)

        # Send everything to PACS
        pacs_config = self.config.get("pacs", {})
        sender = DicomSender(
            local_ae=self.ae_title,
            remote_ae=pacs_config.get("remote_ae_title", "GEPACS"),
            remote_host=pacs_config.get("remote_host", "localhost"),
            remote_port=pacs_config.get("remote_port", 4006),
        )

        sender.send_dataset(sr)
        for sc in annotated_sc:
            sender.send_dataset(sc)
        for gsps in gsps_list:
            sender.send_dataset(gsps)

    def start(self):
        """Start the DICOM SCP server."""
        handlers = [
            (evt.EVT_C_STORE, self._handle_store),
            (evt.EVT_RELEASED, self._handle_release),
        ]

        logger.info(f"Starting DICOM SCP on port {self.port} (AE: {self.ae_title})")
        logger.info("Waiting for incoming CT studies from PACS...")

        self.ae.start_server(
            ("0.0.0.0", self.port),
            evt_handlers=handlers,
            block=True,
        )

    def stop(self):
        """Shutdown the SCP server."""
        self.ae.shutdown()


class DicomSender:
    """DICOM C-STORE SCU for sending results back to PACS."""

    def __init__(
        self,
        local_ae: str = "LUNG_SCREEN_AI",
        remote_ae: str = "GEPACS",
        remote_host: str = "localhost",
        remote_port: int = 4006,
    ):
        self.local_ae = local_ae
        self.remote_ae = remote_ae
        self.remote_host = remote_host
        self.remote_port = remote_port

    def send_dataset(self, dataset: Dataset) -> bool:
        """Send a DICOM dataset to the remote PACS.

        Automatically negotiates the correct presentation context based on
        the dataset's SOPClassUID (SR, Secondary Capture, GSPS, etc.).

        Args:
            dataset: pydicom Dataset to send.

        Returns:
            True if send was successful.
        """
        ae = AE(ae_title=self.local_ae)

        # Add the presentation context matching this dataset's SOP Class
        sop_class = str(dataset.SOPClassUID)
        ae.add_requested_context(sop_class)

        assoc = ae.associate(
            self.remote_host,
            self.remote_port,
            ae_title=self.remote_ae,
        )

        if assoc.is_established:
            status = assoc.send_c_store(dataset)
            assoc.release()

            modality = getattr(dataset, "Modality", "??")
            if status and status.Status == 0x0000:
                logger.info(f"Successfully sent {modality} to PACS")
                return True
            else:
                logger.error(f"Failed to send {modality} to PACS: {status}")
                return False
        else:
            logger.error(
                f"Could not connect to PACS at {self.remote_host}:{self.remote_port}"
            )
            return False


def create_nodule_sr(result: ScanResult, ref_dicom: Dataset) -> FileDataset:
    """Create a DICOM Structured Report containing nodule findings.

    Generates a Comprehensive SR (TID 1500 - Measurement Report) with
    nodule locations, sizes, and Lung-RADS categorization.

    Args:
        result: ScanResult from the detector.
        ref_dicom: Reference DICOM dataset for patient/study info.

    Returns:
        FileDataset ready to be sent via C-STORE.
    """
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Create the SR dataset
    ds = FileDataset(
        filename_or_obj="sr.dcm",
        dataset=Dataset(),
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    # Copy patient and study info from reference
    ds.PatientName = getattr(ref_dicom, "PatientName", "")
    ds.PatientID = getattr(ref_dicom, "PatientID", "")
    ds.PatientBirthDate = getattr(ref_dicom, "PatientBirthDate", "")
    ds.PatientSex = getattr(ref_dicom, "PatientSex", "")
    ds.StudyInstanceUID = getattr(ref_dicom, "StudyInstanceUID", generate_uid())
    ds.StudyDate = getattr(ref_dicom, "StudyDate", "")
    ds.StudyTime = getattr(ref_dicom, "StudyTime", "")
    ds.StudyDescription = getattr(ref_dicom, "StudyDescription", "CT CHEST")
    ds.AccessionNumber = getattr(ref_dicom, "AccessionNumber", "")
    ds.ReferringPhysicianName = getattr(ref_dicom, "ReferringPhysicianName", "")

    # SR-specific attributes
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.Modality = "SR"
    ds.Manufacturer = "LungScreenerAI"
    ds.SeriesDescription = "AI Lung Nodule Detection Report"
    ds.SeriesNumber = 999
    ds.InstanceNumber = 1
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"

    # Concept name: Imaging Measurement Report
    ds.ConceptNameCodeSequence = [_code("126000", "DCM", "Imaging Measurement Report")]

    # Build content tree
    content = []

    # Language of Content
    lang_item = Dataset()
    lang_item.RelationshipType = "HAS CONCEPT MOD"
    lang_item.ValueType = "CODE"
    lang_item.ConceptNameCodeSequence = [_code("121049", "DCM", "Language of Content Item and Descendants")]
    lang_item.ConceptCodeSequence = [_code("eng", "RFC5646", "English")]
    content.append(lang_item)

    # Observation context - device
    device_item = Dataset()
    device_item.RelationshipType = "HAS OBS CONTEXT"
    device_item.ValueType = "CODE"
    device_item.ConceptNameCodeSequence = [_code("121005", "DCM", "Observer Type")]
    device_item.ConceptCodeSequence = [_code("121007", "DCM", "Device")]
    content.append(device_item)

    # Overall Lung-RADS
    rads_item = Dataset()
    rads_item.RelationshipType = "CONTAINS"
    rads_item.ValueType = "TEXT"
    rads_item.ConceptNameCodeSequence = [_code("LUNGRADS", "99LOCAL", "Lung-RADS Category")]
    rads_item.TextValue = result.lung_rads_overall or "1"
    content.append(rads_item)

    # Summary
    summary_item = Dataset()
    summary_item.RelationshipType = "CONTAINS"
    summary_item.ValueType = "TEXT"
    summary_item.ConceptNameCodeSequence = [_code("121077", "DCM", "Conclusion")]
    summary_item.TextValue = (
        f"AI screening detected {len(result.findings)} nodule(s). "
        f"Overall Lung-RADS: {result.lung_rads_overall or '1'}. "
        "This is a computer-aided detection result and should be "
        "reviewed by a qualified radiologist."
    )
    content.append(summary_item)

    # Individual findings
    for i, finding in enumerate(result.findings, 1):
        finding_container = Dataset()
        finding_container.RelationshipType = "CONTAINS"
        finding_container.ValueType = "CONTAINER"
        finding_container.ContinuityOfContent = "SEPARATE"
        finding_container.ConceptNameCodeSequence = [
            _code("121071", "DCM", "Finding")
        ]

        finding_content = []

        # Nodule type
        type_item = Dataset()
        type_item.RelationshipType = "CONTAINS"
        type_item.ValueType = "CODE"
        type_item.ConceptNameCodeSequence = [_code("121071", "DCM", "Finding")]
        type_item.ConceptCodeSequence = [
            _code("RID3875", "RADLEX", "Pulmonary nodule")
        ]
        finding_content.append(type_item)

        # Size
        size_item = Dataset()
        size_item.RelationshipType = "CONTAINS"
        size_item.ValueType = "NUM"
        size_item.ConceptNameCodeSequence = [_code("121211", "DCM", "Path Length")]
        measured = Dataset()
        measured.NumericValue = f"{finding.diameter_mm:.1f}"
        measured.MeasurementUnitsCodeSequence = [_code("mm", "UCUM", "mm")]
        size_item.MeasuredValueSequence = [measured]
        finding_content.append(size_item)

        # Confidence
        conf_item = Dataset()
        conf_item.RelationshipType = "CONTAINS"
        conf_item.ValueType = "NUM"
        conf_item.ConceptNameCodeSequence = [
            _code("CONFIDENCE", "99LOCAL", "Detection Confidence")
        ]
        conf_measured = Dataset()
        conf_measured.NumericValue = f"{finding.confidence:.3f}"
        conf_measured.MeasurementUnitsCodeSequence = [_code("%", "UCUM", "percent")]
        conf_item.MeasuredValueSequence = [conf_measured]
        finding_content.append(conf_item)

        # Lung-RADS for this finding
        rads = Dataset()
        rads.RelationshipType = "CONTAINS"
        rads.ValueType = "TEXT"
        rads.ConceptNameCodeSequence = [
            _code("LUNGRADS", "99LOCAL", "Lung-RADS Category")
        ]
        rads.TextValue = finding.lung_rads
        finding_content.append(rads)

        # Location coordinates
        loc_item = Dataset()
        loc_item.RelationshipType = "CONTAINS"
        loc_item.ValueType = "TEXT"
        loc_item.ConceptNameCodeSequence = [
            _code("121230", "DCM", "Location of Measurement")
        ]
        loc_item.TextValue = f"({finding.x:.1f}, {finding.y:.1f}, {finding.z:.1f}) mm"
        finding_content.append(loc_item)

        finding_container.ContentSequence = Sequence(finding_content)
        content.append(finding_container)

    ds.ContentSequence = Sequence(content)

    return ds


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    """Helper to create a coded concept."""
    code = Dataset()
    code.CodeValue = value
    code.CodingSchemeDesignator = scheme
    code.CodeMeaning = meaning
    return code
