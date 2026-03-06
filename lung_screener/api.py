"""Web API for the lung nodule AI viewer.

Provides REST endpoints for DICOM upload, inference, and DICOM file serving.
The FastAPI app is started by the ``viewer`` CLI command.
"""

import asyncio
import logging
import os
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional

import pydicom
import SimpleITK as sitk
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="Lung Nodule AI Viewer", version="1.0.0")

# Serve bundled JS/CSS vendor files at /static/vendor — no CDN needed
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# In-memory study registry {study_id: study_dict}
_studies: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# Configured by the viewer CLI command before uvicorn starts
UPLOAD_DIR = Path(os.environ.get("LUNG_VIEWER_DIR", "/tmp/lung_viewer"))
_detector = None  # NoduleDetector instance


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "viewer.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.post("/api/studies/upload")
async def upload_study(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Accept DICOM files (individual .dcm or .zip archive) and queue inference."""
    study_id = str(uuid.uuid4())
    study_dir = UPLOAD_DIR / study_id
    dicom_dir = study_dir / "dicom"
    dicom_dir.mkdir(parents=True)

    for f in files:
        fname = f.filename or f"file_{uuid.uuid4().hex}"
        content = await f.read()

        if fname.lower().endswith(".zip"):
            tmp_zip = study_dir / fname
            tmp_zip.write_bytes(content)
            with zipfile.ZipFile(tmp_zip) as zf:
                for member in zf.namelist():
                    if not member.endswith("/"):
                        target = dicom_dir / Path(member).name
                        target.write_bytes(zf.read(member))
        else:
            (dicom_dir / Path(fname).name).write_bytes(content)

    series_map = _organize_dicom(dicom_dir)
    if not series_map:
        raise HTTPException(400, "No valid DICOM files found in upload")

    _studies[study_id] = {
        "id": study_id,
        "status": "queued",
        "created_at": time.time(),
        "dicom_dir": str(dicom_dir),
        "series": series_map,
        "results": None,
        "report": None,
        "error": None,
    }

    # Run inference in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(loop.run_in_executor(_executor, _run_inference, study_id))

    return {"study_id": study_id, "series": series_map}


def _organize_dicom(dicom_dir: Path) -> dict:
    """Read DICOM headers and group files by SeriesInstanceUID."""
    series_map: dict[str, dict] = {}

    for f in sorted(dicom_dir.iterdir()):
        if f.is_dir():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            instance_uid = str(getattr(ds, "SOPInstanceUID", f.stem))
            instance_num = int(getattr(ds, "InstanceNumber", 0))
            slice_loc = float(getattr(ds, "SliceLocation", 0.0))

            if series_uid not in series_map:
                series_map[series_uid] = {
                    "series_uid": series_uid,
                    "description": str(getattr(ds, "SeriesDescription", "CT Series")),
                    "modality": str(getattr(ds, "Modality", "CT")),
                    "patient_name": str(getattr(ds, "PatientName", "Anonymous")),
                    "patient_id": str(getattr(ds, "PatientID", "")),
                    "study_date": str(getattr(ds, "StudyDate", "")),
                    "study_description": str(getattr(ds, "StudyDescription", "")),
                    "instances": [],
                }

            series_map[series_uid]["instances"].append({
                "instance_uid": instance_uid,
                "instance_number": instance_num,
                "slice_location": slice_loc,
                "filename": f.name,
            })
        except Exception:
            continue

    for s in series_map.values():
        s["instances"].sort(key=lambda x: (x["instance_number"], x["slice_location"]))

    return series_map


def _run_inference(study_id: str) -> None:
    """Run model inference synchronously in a thread pool worker."""
    study = _studies.get(study_id)
    if not study:
        return

    study["status"] = "processing"
    try:
        if _detector is None:
            raise RuntimeError("Model not loaded — start the viewer with --checkpoint")

        from .preprocessing import load_dicom_series

        dicom_dir = study["dicom_dir"]
        image: sitk.Image = load_dicom_series(dicom_dir)
        result = _detector.predict_scan(image)

        # Augment each finding with voxel coordinates for the viewer overlay
        findings = []
        for finding in result.findings:
            d = finding.to_dict()
            try:
                world_pt = (finding.x, finding.y, finding.z)
                voxel = image.TransformPhysicalPointToIndex(world_pt)
                d["voxel_x"] = voxel[0]   # column
                d["voxel_y"] = voxel[1]   # row
                d["voxel_z"] = voxel[2]   # slice (0-based)
            except Exception:
                d["voxel_x"] = 0
                d["voxel_y"] = 0
                d["voxel_z"] = max(0, finding.image_number - 1)
            findings.append(d)

        study["results"] = {
            "lung_rads_overall": result.lung_rads_overall or "1",
            "num_findings": len(findings),
            "findings": findings,
        }
        study["report"] = result.dictation()
        study["status"] = "complete"

    except Exception as e:
        logger.exception("Inference failed for study %s", study_id)
        study["status"] = "error"
        study["error"] = str(e)


@app.get("/api/studies/{study_id}/status")
async def get_status(study_id: str):
    study = _studies.get(study_id)
    if not study:
        raise HTTPException(404, "Study not found")
    return {"status": study["status"], "error": study.get("error")}


@app.get("/api/studies/{study_id}/results")
async def get_results(study_id: str):
    study = _studies.get(study_id)
    if not study:
        raise HTTPException(404, "Study not found")
    if study["status"] != "complete":
        raise HTTPException(202, f"Study not ready — status: {study['status']}")
    return study["results"]


@app.get("/api/studies/{study_id}/report")
async def get_report(study_id: str):
    study = _studies.get(study_id)
    if not study:
        raise HTTPException(404, "Study not found")
    return {"report": study.get("report", ""), "status": study["status"]}


@app.get("/api/studies/{study_id}/series")
async def list_series(study_id: str):
    study = _studies.get(study_id)
    if not study:
        raise HTTPException(404, "Study not found")
    return {"series": list(study["series"].values())}


@app.get("/api/studies/{study_id}/dicom/{filename:path}")
async def get_dicom_file(study_id: str, filename: str):
    """Serve a DICOM file for cornerstoneWADOImageLoader.

    Normalises the file to Explicit VR Little Endian (uncompressed) so that
    the browser-side WADO loader can always decode it, regardless of the
    original transfer syntax.  Falls back to raw bytes on any error.
    """
    study = _studies.get(study_id)
    if not study:
        raise HTTPException(404, "Study not found")

    dicom_dir = Path(study["dicom_dir"])
    file_path = (dicom_dir / filename).resolve()

    # Security: ensure the resolved path stays inside the study's dicom dir
    try:
        file_path.relative_to(dicom_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Forbidden")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    raw = file_path.read_bytes()
    normalised = _normalise_dicom(raw)
    return Response(content=normalised, media_type="application/dicom")


# Transfer syntaxes that cornerstoneWADOImageLoader handles natively
_UNCOMPRESSED_TS = {
    "1.2.840.10008.1.2",    # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",  # Explicit VR Big Endian (rare but parseable)
}


def _normalise_dicom(raw: bytes) -> bytes:
    """Convert *raw* DICOM bytes to Explicit VR Little Endian (uncompressed).

    If the file is already uncompressed or if conversion fails, the original
    bytes are returned unchanged so the browser still has something to try.
    """
    try:
        ds = pydicom.dcmread(BytesIO(raw))
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)

        # Already in a format the WADO loader can handle – serve as-is
        if ts is None or str(ts) in _UNCOMPRESSED_TS:
            return raw

        # Compressed transfer syntax: decompress via pixel_array then rewrite
        arr = ds.pixel_array  # triggers decompression; shape (rows, cols) or (frames, rows, cols)

        # Flatten to 2-D for single-frame files
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        ds.PixelData = arr.tobytes()
        ds.is_implicit_VR = False
        ds.is_little_endian = True

        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = getattr(ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2")
        ds.file_meta.MediaStorageSOPInstanceUID = getattr(ds, "SOPInstanceUID", pydicom.uid.generate_uid())

        buf = BytesIO()
        pydicom.dcmwrite(buf, ds)
        return buf.getvalue()

    except Exception:
        logger.debug("DICOM normalisation failed, serving raw bytes", exc_info=True)
        return raw
