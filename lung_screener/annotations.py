"""Generate visual annotations for radiologist review.

Creates DICOM objects that mark nodule locations directly on CT images:

1. Secondary Capture (SC): Rendered images with circles and labels burned in.
   Works on every PACS viewer, even ones that don't support overlays.

2. Grayscale Softcopy Presentation State (GSPS): Native PACS annotations
   that overlay circles/text on the original CT without modifying the pixels.
   The radiologist can toggle them on/off in the viewer.
"""

import logging
from datetime import datetime

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from .inference import NoduleFinding, ScanResult

logger = logging.getLogger(__name__)

# Window/level for lung CT display
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500


# ---------------------------------------------------------------------------
# Secondary Capture — burned-in annotations
# ---------------------------------------------------------------------------

def create_annotated_slices(
    result: ScanResult,
    dicom_files: list[pydicom.Dataset],
) -> list[FileDataset]:
    """Create Secondary Capture images with nodule markers burned in.

    For each finding, renders a circle and label on the closest axial slice.
    These appear as a new series in the PACS viewer.

    Args:
        result: Detection result with findings.
        dicom_files: Original CT DICOM datasets (one per slice), sorted by
            ImagePositionPatient z.

    Returns:
        List of annotated Secondary Capture FileDatasets ready for C-STORE.
    """
    if not result.findings or not dicom_files:
        return []

    # Sort slices by z position
    slices = sorted(dicom_files, key=lambda d: float(d.ImagePositionPatient[2]))
    z_positions = np.array([float(d.ImagePositionPatient[2]) for d in slices])

    # Map each finding to its nearest slice
    slice_findings: dict[int, list[NoduleFinding]] = {}
    for finding in result.findings:
        slice_idx = int(np.argmin(np.abs(z_positions - finding.z)))
        slice_findings.setdefault(slice_idx, []).append(finding)

    # Generate a new series UID for all annotated slices
    series_uid = generate_uid()
    sc_datasets = []

    for slice_idx, findings in slice_findings.items():
        ds = slices[slice_idx]
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept to get HU
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = pixel_array * slope + intercept

        # Apply lung window
        img = _apply_window(hu, LUNG_WINDOW_CENTER, LUNG_WINDOW_WIDTH)

        # Convert to RGB so we can draw colored annotations
        rgb = np.stack([img, img, img], axis=-1)

        # Draw annotations for each finding on this slice
        for finding in findings:
            _draw_finding_marker(rgb, ds, finding)

        # Create Secondary Capture
        sc = _create_sc_dataset(rgb, ds, series_uid, len(sc_datasets) + 1, result)
        sc_datasets.append(sc)

    logger.info(
        f"Created {len(sc_datasets)} annotated Secondary Capture image(s) "
        f"for {len(result.findings)} finding(s)"
    )
    return sc_datasets


def _apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply window/level to HU values, returning 0-255 uint8."""
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip((hu - lower) / (upper - lower) * 255, 0, 255)
    return img.astype(np.uint8)


def _world_to_pixel(ds: pydicom.Dataset, x: float, y: float) -> tuple[int, int]:
    """Convert world coordinates (x, y) to pixel (col, row) on a DICOM slice."""
    ipp = [float(v) for v in ds.ImagePositionPatient]
    iop = [float(v) for v in ds.ImageOrientationPatient]
    spacing = [float(v) for v in ds.PixelSpacing]

    # Row and column direction cosines
    row_cos = np.array(iop[0:3])
    col_cos = np.array(iop[3:6])

    # Offset from image origin
    dx = x - ipp[0]
    dy = y - ipp[1]
    dz = 0  # We already matched the slice

    delta = np.array([dx, dy, dz])
    col = np.dot(delta, row_cos) / spacing[1]
    row = np.dot(delta, col_cos) / spacing[0]

    return int(round(col)), int(round(row))


def _draw_finding_marker(
    rgb: np.ndarray,
    ds: pydicom.Dataset,
    finding: NoduleFinding,
):
    """Draw a circle and label on the RGB image at the finding location."""
    col, row = _world_to_pixel(ds, finding.x, finding.y)
    rows, cols = rgb.shape[:2]

    # Radius from nodule diameter (in pixels)
    pixel_spacing = float(ds.PixelSpacing[0])
    radius_px = max(int(finding.diameter_mm / pixel_spacing / 2), 6)
    # Add padding so the circle is visible around the nodule
    draw_radius = radius_px + 4

    # Draw circle (bright green: 0, 255, 0)
    color = np.array([0, 255, 0], dtype=np.uint8)
    _draw_circle(rgb, row, col, draw_radius, color, thickness=2)

    # Draw crosshair lines extending from the circle
    line_len = draw_radius + 8
    _draw_line(rgb, row, col - line_len, row, col - draw_radius - 2, color)
    _draw_line(rgb, row, col + draw_radius + 2, row, col + line_len, color)
    _draw_line(rgb, row - line_len, col, row - draw_radius - 2, col, color)
    _draw_line(rgb, row + draw_radius + 2, col, row + line_len, col, color)

    # Draw label below the circle
    label = f"{finding.diameter_mm:.0f}mm LR-{finding.lung_rads}"
    label_row = min(row + draw_radius + 14, rows - 10)
    label_col = max(col - len(label) * 3, 5)
    _draw_text_simple(rgb, label_row, label_col, label, color)


def _draw_circle(
    img: np.ndarray, cy: int, cx: int, radius: int,
    color: np.ndarray, thickness: int = 2,
):
    """Draw a circle using midpoint algorithm."""
    h, w = img.shape[:2]
    for angle_deg in range(360):
        angle = np.radians(angle_deg)
        for t in range(thickness):
            r = radius + t
            py = int(round(cy + r * np.sin(angle)))
            px = int(round(cx + r * np.cos(angle)))
            if 0 <= py < h and 0 <= px < w:
                img[py, px] = color


def _draw_line(
    img: np.ndarray,
    r0: int, c0: int, r1: int, c1: int,
    color: np.ndarray,
):
    """Draw a 1-pixel line using Bresenham's algorithm."""
    h, w = img.shape[:2]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        if 0 <= r0 < h and 0 <= c0 < w:
            img[r0, c0] = color
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr


def _draw_text_simple(
    img: np.ndarray, row: int, col: int, text: str,
    color: np.ndarray,
):
    """Render text as simple block characters (no font dependency).

    Each character is drawn as a small 5x3 block pattern. Not pretty,
    but guaranteed to work without Pillow/freetype.
    """
    # Minimal 5x3 bitmap font for digits, uppercase, and common symbols
    font = _get_mini_font()
    h, w = img.shape[:2]

    for i, ch in enumerate(text.upper()):
        glyph = font.get(ch, font.get("?", []))
        cx = col + i * 5
        for gy, glyph_row in enumerate(glyph):
            for gx, pixel in enumerate(glyph_row):
                if pixel:
                    py, px = row + gy, cx + gx
                    if 0 <= py < h and 0 <= px < w:
                        img[py, px] = color


def _get_mini_font() -> dict[str, list[list[int]]]:
    """Minimal 5-row x 3-col bitmap font for annotation labels."""
    return {
        "0": [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
        "1": [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
        "2": [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
        "3": [[1,1,1],[0,0,1],[1,1,1],[0,0,1],[1,1,1]],
        "4": [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
        "5": [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
        "6": [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
        "7": [[1,1,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0]],
        "8": [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
        "9": [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
        "A": [[0,1,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
        "B": [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,1,0]],
        "L": [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1]],
        "M": [[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1]],
        "R": [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,0,1]],
        "-": [[0,0,0],[0,0,0],[1,1,1],[0,0,0],[0,0,0]],
        ".": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,0]],
        " ": [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    }


def _create_sc_dataset(
    rgb: np.ndarray,
    ref_ds: pydicom.Dataset,
    series_uid: str,
    instance_number: int,
    result: ScanResult,
) -> FileDataset:
    """Wrap an RGB annotated image as a DICOM Secondary Capture."""
    sop_uid = generate_uid()

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # SC
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset("sc.dcm", Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient / Study — same as original
    ds.PatientName = getattr(ref_ds, "PatientName", "")
    ds.PatientID = getattr(ref_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(ref_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(ref_ds, "PatientSex", "")
    ds.StudyInstanceUID = getattr(ref_ds, "StudyInstanceUID", generate_uid())
    ds.StudyDate = getattr(ref_ds, "StudyDate", "")
    ds.StudyTime = getattr(ref_ds, "StudyTime", "")
    ds.AccessionNumber = getattr(ref_ds, "AccessionNumber", "")

    # Series — new series for annotations
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = "AI Nodule Detection - Annotated"
    ds.SeriesNumber = 998
    ds.Modality = "OT"  # Other

    # Instance
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")

    # Copy spatial info from original slice
    if hasattr(ref_ds, "ImagePositionPatient"):
        ds.ImagePositionPatient = ref_ds.ImagePositionPatient
    if hasattr(ref_ds, "ImageOrientationPatient"):
        ds.ImageOrientationPatient = ref_ds.ImageOrientationPatient
    if hasattr(ref_ds, "SliceLocation"):
        ds.SliceLocation = ref_ds.SliceLocation
    if hasattr(ref_ds, "FrameOfReferenceUID"):
        ds.FrameOfReferenceUID = ref_ds.FrameOfReferenceUID

    # Pixel data (RGB)
    ds.Rows = rgb.shape[0]
    ds.Columns = rgb.shape[1]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = rgb.tobytes()

    return ds


# ---------------------------------------------------------------------------
# GSPS Presentation State — native PACS overlay annotations
# ---------------------------------------------------------------------------

def create_gsps(
    result: ScanResult,
    dicom_files: list[pydicom.Dataset],
) -> list[FileDataset]:
    """Create GSPS Presentation State objects with nodule annotations.

    GSPS (Grayscale Softcopy Presentation State) lets the PACS viewer
    draw circles and text on top of the original CT images without
    modifying any pixels. The radiologist can toggle annotations on/off.

    Args:
        result: Detection result.
        dicom_files: Original CT DICOM datasets sorted by z.

    Returns:
        List of GSPS FileDatasets (one per annotated slice).
    """
    if not result.findings or not dicom_files:
        return []

    slices = sorted(dicom_files, key=lambda d: float(d.ImagePositionPatient[2]))
    z_positions = np.array([float(d.ImagePositionPatient[2]) for d in slices])

    # Map findings to slices
    slice_findings: dict[int, list[NoduleFinding]] = {}
    for finding in result.findings:
        slice_idx = int(np.argmin(np.abs(z_positions - finding.z)))
        slice_findings.setdefault(slice_idx, []).append(finding)

    gsps_list = []

    for slice_idx, findings in slice_findings.items():
        ds = slices[slice_idx]
        gsps = _create_gsps_dataset(ds, findings, result)
        gsps_list.append(gsps)

    logger.info(f"Created {len(gsps_list)} GSPS Presentation State(s)")
    return gsps_list


def _create_gsps_dataset(
    ref_ds: pydicom.Dataset,
    findings: list[NoduleFinding],
    result: ScanResult,
) -> FileDataset:
    """Build a single GSPS dataset for one slice with its findings."""
    sop_uid = generate_uid()

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.11.1"  # GSPS
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset("gsps.dcm", Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient / Study
    ds.PatientName = getattr(ref_ds, "PatientName", "")
    ds.PatientID = getattr(ref_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(ref_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(ref_ds, "PatientSex", "")
    ds.StudyInstanceUID = getattr(ref_ds, "StudyInstanceUID", generate_uid())
    ds.StudyDate = getattr(ref_ds, "StudyDate", "")
    ds.AccessionNumber = getattr(ref_ds, "AccessionNumber", "")

    # Series
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = "AI Nodule Annotations"
    ds.SeriesNumber = 997
    ds.Modality = "PR"  # Presentation State

    # Instance
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.11.1"
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = 1
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ds.ContentLabel = "AI_NODULE"
    ds.ContentDescription = "Lung Screener AI - Detected Nodules"
    ds.ContentCreatorName = "LungScreenerAI"
    ds.PresentationCreationDate = ds.ContentDate
    ds.PresentationCreationTime = ds.ContentTime

    # Reference the original CT image
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = ref_ds.SeriesInstanceUID

    ref_image = Dataset()
    ref_image.ReferencedSOPClassUID = ref_ds.SOPClassUID
    ref_image.ReferencedSOPInstanceUID = ref_ds.SOPInstanceUID
    ref_series.ReferencedImageSequence = Sequence([ref_image])

    ds.ReferencedSeriesSequence = Sequence([ref_series])

    # Display window for lung CT
    ds.WindowCenter = str(LUNG_WINDOW_CENTER)
    ds.WindowWidth = str(LUNG_WINDOW_WIDTH)

    # Graphic Annotation Sequence — the actual markings
    graphic_annotations = []

    for finding in findings:
        col, row = _world_to_pixel(ref_ds, finding.x, finding.y)
        pixel_spacing = float(ref_ds.PixelSpacing[0])
        radius_px = max(finding.diameter_mm / pixel_spacing / 2, 6) + 4

        annotation = Dataset()

        # Reference which image this annotation applies to
        ref_img = Dataset()
        ref_img.ReferencedSOPClassUID = ref_ds.SOPClassUID
        ref_img.ReferencedSOPInstanceUID = ref_ds.SOPInstanceUID
        annotation.ReferencedImageSequence = Sequence([ref_img])

        graphic_objects = []
        text_objects = []

        # Circle around the nodule
        circle = Dataset()
        circle.GraphicAnnotationUnits = "PIXEL"
        circle.GraphicDimensions = 2
        # CIRCLE: center point + a point on the circumference
        circle.NumberOfGraphicPoints = 2
        circle.GraphicData = [
            float(col), float(row),
            float(col + radius_px), float(row),
        ]
        circle.GraphicType = "CIRCLE"
        circle.GraphicFilled = "N"
        graphic_objects.append(circle)

        # Crosshair lines
        line_len = radius_px + 8
        for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            line = Dataset()
            line.GraphicAnnotationUnits = "PIXEL"
            line.GraphicDimensions = 2
            line.NumberOfGraphicPoints = 2
            start_gap = radius_px + 2
            line.GraphicData = [
                float(col + dc * start_gap), float(row + dr * start_gap),
                float(col + dc * line_len), float(row + dr * line_len),
            ]
            line.GraphicType = "POLYLINE"
            line.GraphicFilled = "N"
            graphic_objects.append(line)

        # Text label
        label = f"{finding.diameter_mm:.0f}mm LR-{finding.lung_rads} ({finding.confidence:.0%})"
        text_obj = Dataset()
        text_obj.UnformattedTextValue = label
        text_obj.TextObjectAnchorPointAnnotationUnits = "PIXEL"
        text_obj.AnchorPoint = [float(col), float(row + radius_px + 12)]
        text_obj.AnchorPointVisibility = "Y"
        text_objects.append(text_obj)

        annotation.GraphicObjectSequence = Sequence(graphic_objects)
        annotation.TextObjectSequence = Sequence(text_objects)
        graphic_annotations.append(annotation)

    ds.GraphicAnnotationSequence = Sequence(graphic_annotations)

    # Graphic Layer Sequence
    layer = Dataset()
    layer.GraphicLayer = "FINDINGS"
    layer.GraphicLayerOrder = 1
    layer.GraphicLayerDescription = "AI-detected lung nodules"
    ds.GraphicLayerSequence = Sequence([layer])

    return ds
