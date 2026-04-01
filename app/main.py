from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from app.dicom_utils import load_study_volume, normalize_for_display
from app.model import LungNoduleModel
from app.settings import API_TITLE, DATA_ROOT
from app.storage import create_study_dir, list_studies, save_uploaded_files, ensure_data_root

app = FastAPI(title=API_TITLE)
model = LungNoduleModel()


@app.on_event("startup")
def on_startup() -> None:
    ensure_data_root()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/studies")
def studies() -> dict:
    return {"studies": list_studies()}


@app.post("/studies/upload")
async def upload_study(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    study_dir = create_study_dir()
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        filename = f.filename or "unknown"
        blob = await f.read()
        if filename.lower().endswith(".zip"):
            with ZipFile(BytesIO(blob)) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(".dcm"):
                        payloads.append((Path(info.filename).name, zf.read(info)))
        else:
            payloads.append((filename, blob))
    count = save_uploaded_files(study_dir, payloads)
    if count == 0:
        raise HTTPException(status_code=400, detail="No .dcm files uploaded")

    return {"study_id": study_dir.name, "saved_files": count}


@app.get("/studies/{study_id}/metadata")
def study_metadata(study_id: str) -> dict:
    study_dir = DATA_ROOT / study_id
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found")
    _, meta = load_study_volume(study_dir)
    return meta


@app.get("/studies/{study_id}/slice/{slice_index}")
def slice_png(study_id: str, slice_index: int, center: int = -600, width: int = 1500):
    study_dir = DATA_ROOT / study_id
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found")

    volume, _ = load_study_volume(study_dir)
    if slice_index < 0 or slice_index >= volume.shape[0]:
        raise HTTPException(status_code=400, detail="Invalid slice index")

    img_array = normalize_for_display(volume[slice_index], window_center=center, window_width=width)
    image = Image.fromarray(img_array)
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/studies/{study_id}/infer")
def infer(study_id: str):
    study_dir = DATA_ROOT / study_id
    if not study_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found")

    volume, meta = load_study_volume(study_dir)
    result = model.predict(volume)
    return JSONResponse(
        {
            "study_id": study_id,
            "metadata": meta,
            "model": model.model_name,
            "risk_score": round(result.risk_score, 4),
            "finding_count": result.finding_count,
            "findings": result.findings,
            "notes": result.notes,
        }
    )
