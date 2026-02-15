"""Local web UI for annotating CT scans.

Serves a browser-based tool where you can:
- Drag-and-drop or browse to import DICOM folders
- Scroll through axial slices
- Click on a nodule to mark its location
- Drag to estimate diameter
- See all annotations overlaid on the slices
- Mark scans as negative (no nodules)
- Prepare training data and kick off training

Launch with:
    lung-screener annotate --port 8888
"""

import io
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .data_manager import DataManager
from .preprocessing import CTPreprocessor

logger = logging.getLogger(__name__)

# Lazy-initialized globals (set by create_app)
_dm: DataManager | None = None
_config: dict = {}

# Cache for loaded volume data: series_uid -> (hu_volume, spacing, origin, sitk_image)
_volume_cache: dict[str, tuple] = {}

LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500


def create_app(data_dir: str | Path, config: dict) -> FastAPI:
    global _dm, _config
    _dm = DataManager(data_dir, config)
    _config = config

    app = FastAPI(title="Lung Screener AI - Annotation Tool")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(_router)
    return app


# ---- FastAPI routes ----

from fastapi import APIRouter

_router = APIRouter()


@_router.get("/", response_class=HTMLResponse)
def index():
    return _FRONTEND_HTML


@_router.get("/api/scans")
def list_scans():
    return _dm.list_scans()


@_router.get("/api/scans/{series_uid}")
def get_scan(series_uid: str):
    if series_uid not in _dm.manifest["scans"]:
        raise HTTPException(404, "Scan not found")
    return _dm.manifest["scans"][series_uid]


@_router.get("/api/scans/{series_uid}/slice/{slice_idx}")
def get_slice(series_uid: str, slice_idx: int):
    """Return a PNG image of one axial slice with lung windowing."""
    vol_data = _load_volume_cached(series_uid)
    if vol_data is None:
        raise HTTPException(404, "Volume not found")

    hu_volume, spacing, origin, _ = vol_data

    if slice_idx < 0 or slice_idx >= hu_volume.shape[0]:
        raise HTTPException(400, f"Slice index out of range (0-{hu_volume.shape[0]-1})")

    sl = hu_volume[slice_idx]
    img = _apply_window(sl, LUNG_WINDOW_CENTER, LUNG_WINDOW_WIDTH)
    png_bytes = _encode_png(img)

    return Response(content=png_bytes, media_type="image/png")


@_router.get("/api/scans/{series_uid}/info")
def get_volume_info(series_uid: str):
    """Return spatial metadata for coordinate mapping."""
    vol_data = _load_volume_cached(series_uid)
    if vol_data is None:
        raise HTTPException(404, "Volume not found")

    hu_volume, spacing, origin, _ = vol_data
    return {
        "num_slices": hu_volume.shape[0],
        "height": hu_volume.shape[1],
        "width": hu_volume.shape[2],
        "spacing": list(spacing),
        "origin": list(origin),
    }


class AnnotationRequest(BaseModel):
    coordX: float
    coordY: float
    coordZ: float
    diameter_mm: float
    note: str = ""


@_router.post("/api/scans/{series_uid}/annotate")
def add_annotation(series_uid: str, req: AnnotationRequest):
    try:
        ann = _dm.annotate(
            series_uid, req.coordX, req.coordY, req.coordZ,
            req.diameter_mm, note=req.note,
        )
        return ann
    except ValueError as e:
        raise HTTPException(400, str(e))


@_router.post("/api/scans/{series_uid}/mark-negative")
def mark_negative(series_uid: str):
    try:
        _dm.mark_negative(series_uid)
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@_router.delete("/api/scans/{series_uid}/annotations/{ann_idx}")
def delete_annotation(series_uid: str, ann_idx: int):
    if series_uid not in _dm.manifest["scans"]:
        raise HTTPException(404, "Scan not found")
    scan = _dm.manifest["scans"][series_uid]
    annotations = scan.get("annotations", [])
    if ann_idx < 0 or ann_idx >= len(annotations):
        raise HTTPException(400, "Invalid annotation index")
    annotations.pop(ann_idx)
    if not annotations:
        scan["status"] = "imported"
    _dm._save_manifest()
    return {"status": "ok"}


class ImportRequest(BaseModel):
    path: str
    label: str = ""


@_router.post("/api/import")
def import_scan(req: ImportRequest):
    try:
        scan = _dm.import_dicom(req.path, label=req.label)
        return scan
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@_router.post("/api/prepare")
def prepare_training_data():
    try:
        result_dir = _dm.prepare()
        return {"dataset_dir": str(result_dir)}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ---- Helpers ----

def _load_volume_cached(series_uid: str):
    if series_uid in _volume_cache:
        return _volume_cache[series_uid]

    if series_uid not in _dm.manifest["scans"]:
        return None

    scan = _dm.manifest["scans"][series_uid]
    mhd_path = _dm.data_dir / scan["mhd_path"]
    if not mhd_path.exists():
        return None

    image = sitk.ReadImage(str(mhd_path))
    preprocessor = CTPreprocessor(_config)
    resampled = sitk.ResampleImageFilter()
    target_spacing = tuple(_config.get("preprocessing", {}).get("target_spacing", [1.0, 1.0, 1.0]))
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resampled.SetOutputSpacing(target_spacing)
    resampled.SetSize(new_size)
    resampled.SetOutputDirection(image.GetDirection())
    resampled.SetOutputOrigin(image.GetOrigin())
    resampled.SetTransform(sitk.Transform())
    resampled.SetInterpolator(sitk.sitkBSpline)
    res_image = resampled.Execute(image)

    hu_volume = sitk.GetArrayFromImage(res_image).astype(np.float32)
    spacing = res_image.GetSpacing()
    origin = res_image.GetOrigin()

    result = (hu_volume, spacing, origin, res_image)
    _volume_cache[series_uid] = result
    return result


def _apply_window(hu_slice: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2
    upper = center + width / 2
    img = np.clip((hu_slice - lower) / (upper - lower) * 255, 0, 255)
    return img.astype(np.uint8)


def _encode_png(gray: np.ndarray) -> bytes:
    """Encode a 2D uint8 array as PNG without Pillow dependency."""
    import struct
    import zlib

    h, w = gray.shape
    raw_data = b""
    for row in gray:
        raw_data += b"\x00" + row.tobytes()

    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    compressed = zlib.compress(raw_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


# ---- Frontend ----

_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lung Screener AI - Annotation Tool</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; }

  /* Sidebar */
  #sidebar { width: 320px; background: #16213e; padding: 16px; overflow-y: auto;
             border-right: 1px solid #0f3460; display: flex; flex-direction: column; }
  #sidebar h1 { font-size: 16px; color: #00d4ff; margin-bottom: 4px; }
  #sidebar .subtitle { font-size: 11px; color: #888; margin-bottom: 16px; }

  .section-title { font-size: 12px; font-weight: 600; color: #aaa; text-transform: uppercase;
                   letter-spacing: 1px; margin: 16px 0 8px; }

  /* Import */
  #import-box { background: #0f3460; border: 2px dashed #00d4ff; border-radius: 8px;
                padding: 16px; text-align: center; cursor: pointer; margin-bottom: 12px; }
  #import-box:hover { background: #1a4a7a; }
  #import-box input { display: none; }
  #import-path { width: 100%; padding: 8px; background: #0d1b36; border: 1px solid #333;
                 border-radius: 4px; color: #fff; font-size: 13px; margin-bottom: 8px; }
  .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;
         font-size: 13px; font-weight: 600; }
  .btn-primary { background: #00d4ff; color: #000; }
  .btn-primary:hover { background: #33dfff; }
  .btn-danger { background: #e74c3c; color: #fff; }
  .btn-danger:hover { background: #ff6b5a; }
  .btn-success { background: #2ecc71; color: #000; }
  .btn-success:hover { background: #4ddb89; }
  .btn-sm { padding: 4px 10px; font-size: 12px; }
  .btn-block { width: 100%; margin-bottom: 8px; }

  /* Scan list */
  .scan-item { background: #0d1b36; border-radius: 6px; padding: 10px; margin-bottom: 6px;
               cursor: pointer; border: 2px solid transparent; transition: border-color 0.15s; }
  .scan-item:hover { border-color: #00d4ff44; }
  .scan-item.active { border-color: #00d4ff; }
  .scan-label { font-weight: 600; font-size: 13px; }
  .scan-meta { font-size: 11px; color: #888; margin-top: 2px; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px;
           font-weight: 600; margin-left: 4px; }
  .badge-imported { background: #555; }
  .badge-annotated { background: #2ecc71; color: #000; }
  .badge-negative { background: #e67e22; color: #000; }

  /* Main viewer */
  #viewer { flex: 1; display: flex; flex-direction: column; }
  #toolbar { background: #16213e; padding: 8px 16px; display: flex; align-items: center;
             gap: 12px; border-bottom: 1px solid #0f3460; }
  #toolbar label { font-size: 12px; color: #aaa; }
  #slice-slider { flex: 1; }
  #slice-info { font-size: 13px; font-family: monospace; min-width: 120px; }

  #canvas-wrap { flex: 1; position: relative; display: flex; align-items: center;
                 justify-content: center; overflow: hidden; background: #000; }
  canvas { cursor: crosshair; }

  /* Annotation panel */
  #ann-panel { width: 280px; background: #16213e; padding: 16px; overflow-y: auto;
               border-left: 1px solid #0f3460; }
  .ann-item { background: #0d1b36; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
              font-size: 12px; position: relative; }
  .ann-item .delete-btn { position: absolute; top: 6px; right: 8px; background: none;
                          border: none; color: #e74c3c; cursor: pointer; font-size: 14px; }
  .ann-coords { font-family: monospace; color: #00d4ff; }
  .ann-size { color: #2ecc71; }

  #diameter-input { width: 80px; padding: 4px 8px; background: #0d1b36; border: 1px solid #333;
                    border-radius: 4px; color: #fff; font-size: 13px; }

  /* Empty state */
  .empty-state { text-align: center; padding: 40px 20px; color: #555; }
  .empty-state h2 { font-size: 18px; margin-bottom: 8px; color: #888; }

  /* Status bar */
  #status-bar { background: #0d1b36; padding: 6px 16px; font-size: 11px; color: #888;
                border-top: 1px solid #0f3460; }

  /* Tooltip */
  #tooltip { position: absolute; background: rgba(0,0,0,0.85); color: #fff; padding: 4px 8px;
             border-radius: 4px; font-size: 11px; pointer-events: none; display: none;
             font-family: monospace; z-index: 10; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Lung Screener AI</h1>
  <div class="subtitle">Annotation Tool</div>

  <div class="section-title">Import Scan</div>
  <input id="import-path" type="text" placeholder="Path to DICOM folder on server...">
  <input id="import-label" type="text" placeholder="Label (optional)" style="width:100%;padding:8px;background:#0d1b36;border:1px solid #333;border-radius:4px;color:#fff;font-size:13px;margin-bottom:8px;">
  <button class="btn btn-primary btn-block" onclick="importScan()">Import DICOM Folder</button>

  <div class="section-title">Scans (<span id="scan-count">0</span>)</div>
  <div id="scan-list"></div>

  <div style="margin-top:auto; padding-top:16px;">
    <button class="btn btn-success btn-block" onclick="prepareData()">Prepare Training Data</button>
  </div>
</div>

<div id="viewer">
  <div id="toolbar">
    <label>Slice:</label>
    <input id="slice-slider" type="range" min="0" max="0" value="0" oninput="loadSlice(this.value)">
    <span id="slice-info">-- / --</span>
    <label style="margin-left:12px;">Diameter (mm):</label>
    <input id="diameter-input" type="number" value="6" min="1" max="50" step="0.5">
    <label style="margin-left:12px;">W/L:</label>
    <select id="window-preset" onchange="changeWindow()" style="background:#0d1b36;color:#fff;border:1px solid #333;padding:4px;border-radius:4px;">
      <option value="-600,1500">Lung</option>
      <option value="40,400">Mediastinum</option>
      <option value="-600,600">Soft Tissue</option>
    </select>
  </div>

  <div id="canvas-wrap">
    <canvas id="ct-canvas"></canvas>
    <div id="tooltip"></div>
  </div>

  <div id="status-bar">Ready. Import a DICOM scan to begin.</div>
</div>

<div id="ann-panel">
  <div class="section-title">Annotations</div>
  <div id="ann-list">
    <div class="empty-state"><p>Select a scan to view annotations</p></div>
  </div>
  <div style="margin-top:12px;">
    <button class="btn btn-danger btn-block btn-sm" onclick="markNegative()" id="mark-neg-btn" style="display:none;">
      Mark as Negative (No Nodules)
    </button>
  </div>
</div>

<script>
const canvas = document.getElementById('ct-canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slice-slider');
const sliceInfo = document.getElementById('slice-info');
const tooltip = document.getElementById('tooltip');

let currentScan = null;
let volumeInfo = null;
let currentSlice = 0;
let sliceImage = null;
let annotations = [];

// ---- API helpers ----
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ---- Scan list ----
async function loadScans() {
  const scans = await api('GET', '/api/scans');
  document.getElementById('scan-count').textContent = scans.length;
  const list = document.getElementById('scan-list');
  list.innerHTML = '';
  for (const s of scans) {
    const div = document.createElement('div');
    div.className = 'scan-item' + (currentScan && currentScan.series_uid === s.series_uid ? ' active' : '');
    const badgeClass = s.status === 'annotated' ? 'badge-annotated' : s.status === 'negative' ? 'badge-negative' : 'badge-imported';
    div.innerHTML = '<div class="scan-label">' + (s.label || s.series_uid.slice(-12)) +
      ' <span class="badge ' + badgeClass + '">' + s.status + '</span></div>' +
      '<div class="scan-meta">' + s.num_slices + ' slices | ' + s.num_annotations + ' annotations</div>';
    div.onclick = () => selectScan(s.series_uid);
    list.appendChild(div);
  }
}

async function selectScan(uid) {
  setStatus('Loading scan...');
  currentScan = await api('GET', '/api/scans/' + uid);
  volumeInfo = await api('GET', '/api/scans/' + uid + '/info');
  annotations = currentScan.annotations || [];

  slider.max = volumeInfo.num_slices - 1;
  slider.value = Math.floor(volumeInfo.num_slices / 2);
  currentSlice = parseInt(slider.value);

  document.getElementById('mark-neg-btn').style.display = 'block';
  await loadSlice(currentSlice);
  renderAnnotations();
  loadScans();
  setStatus('Scan loaded: ' + (currentScan.label || uid.slice(-12)) + ' | Scroll or drag slider to navigate | Click to annotate');
}

async function loadSlice(idx) {
  if (!currentScan) return;
  currentSlice = parseInt(idx);
  slider.value = currentSlice;
  sliceInfo.textContent = (currentSlice + 1) + ' / ' + volumeInfo.num_slices;

  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    sliceImage = img;
    drawFrame();
  };
  img.src = '/api/scans/' + currentScan.series_uid + '/slice/' + currentSlice + '?t=' + Date.now();
}

// ---- Drawing ----
function drawFrame() {
  if (!sliceImage) return;
  ctx.drawImage(sliceImage, 0, 0);

  // Draw annotations on this slice
  if (!volumeInfo) return;
  const spacing = volumeInfo.spacing; // (x, y, z) from SimpleITK
  const origin = volumeInfo.origin;

  for (let i = 0; i < annotations.length; i++) {
    const a = annotations[i];
    // Which slice does this annotation fall on?
    // slice index = (coordZ - origin[2]) / spacing[2]   (SimpleITK z)
    const annSlice = Math.round((a.coordZ - origin[2]) / spacing[2]);
    const sliceDist = Math.abs(annSlice - currentSlice);

    if (sliceDist > 3) continue; // too far, don't show

    // Pixel position on the image
    // numpy array is (z, y, x) but image pixels are (col=x, row=y)
    // pixel_col = (coordX - origin[0]) / spacing[0]
    // pixel_row = (coordY - origin[1]) / spacing[1]
    const px = (a.coordX - origin[0]) / spacing[0];
    const py = (a.coordY - origin[1]) / spacing[1];
    const radiusPx = (a.diameter_mm / 2) / spacing[0];

    const alpha = sliceDist === 0 ? 1.0 : 0.3;
    ctx.strokeStyle = 'rgba(0, 255, 100, ' + alpha + ')';
    ctx.lineWidth = sliceDist === 0 ? 2 : 1;

    // Circle
    ctx.beginPath();
    ctx.arc(px, py, Math.max(radiusPx, 4) + 3, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshair
    const r = Math.max(radiusPx, 4) + 6;
    ctx.beginPath();
    ctx.moveTo(px - r - 4, py); ctx.lineTo(px - r + 2 - 8, py);
    ctx.moveTo(px + r + 4, py); ctx.lineTo(px + r - 2 + 8, py);
    ctx.moveTo(px, py - r - 4); ctx.lineTo(px, py - r + 2 - 8);
    ctx.moveTo(px, py + r + 4); ctx.lineTo(px, py + r - 2 + 8);
    ctx.stroke();

    // Label
    if (sliceDist === 0) {
      ctx.font = '12px monospace';
      ctx.fillStyle = 'rgba(0, 255, 100, 0.9)';
      ctx.fillText(a.diameter_mm.toFixed(0) + 'mm', px + r + 6, py + 4);
    }
  }
}

// ---- Interaction ----
canvas.addEventListener('click', (e) => {
  if (!currentScan || !volumeInfo) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;

  const spacing = volumeInfo.spacing;
  const origin = volumeInfo.origin;

  const worldX = origin[0] + px * spacing[0];
  const worldY = origin[1] + py * spacing[1];
  const worldZ = origin[2] + currentSlice * spacing[2];
  const diameter = parseFloat(document.getElementById('diameter-input').value) || 6;

  addAnnotation(worldX, worldY, worldZ, diameter);
});

canvas.addEventListener('mousemove', (e) => {
  if (!volumeInfo) { tooltip.style.display = 'none'; return; }
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;

  const spacing = volumeInfo.spacing;
  const origin = volumeInfo.origin;
  const wx = (origin[0] + px * spacing[0]).toFixed(1);
  const wy = (origin[1] + py * spacing[1]).toFixed(1);
  const wz = (origin[2] + currentSlice * spacing[2]).toFixed(1);

  tooltip.style.display = 'block';
  tooltip.style.left = (e.clientX - canvas.parentElement.getBoundingClientRect().left + 12) + 'px';
  tooltip.style.top = (e.clientY - canvas.parentElement.getBoundingClientRect().top - 24) + 'px';
  tooltip.textContent = 'x:' + wx + '  y:' + wy + '  z:' + wz + ' mm';
});

canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

// Scroll wheel to navigate slices
document.getElementById('canvas-wrap').addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!volumeInfo) return;
  const newSlice = Math.max(0, Math.min(volumeInfo.num_slices - 1,
    currentSlice + (e.deltaY > 0 ? 1 : -1)));
  if (newSlice !== currentSlice) loadSlice(newSlice);
}, { passive: false });

// ---- Annotations ----
async function addAnnotation(x, y, z, diameter) {
  const ann = await api('POST', '/api/scans/' + currentScan.series_uid + '/annotate', {
    coordX: x, coordY: y, coordZ: z, diameter_mm: diameter
  });
  annotations.push(ann);
  drawFrame();
  renderAnnotations();
  loadScans();
  setStatus('Annotation added at (' + x.toFixed(1) + ', ' + y.toFixed(1) + ', ' + z.toFixed(1) + ') mm');
}

async function deleteAnnotation(idx) {
  await api('DELETE', '/api/scans/' + currentScan.series_uid + '/annotations/' + idx);
  annotations.splice(idx, 1);
  drawFrame();
  renderAnnotations();
  loadScans();
}

function renderAnnotations() {
  const list = document.getElementById('ann-list');
  if (!annotations.length) {
    list.innerHTML = '<div class="empty-state"><p>Click on a nodule in the scan to annotate it</p></div>';
    return;
  }
  list.innerHTML = '';
  for (let i = 0; i < annotations.length; i++) {
    const a = annotations[i];
    const div = document.createElement('div');
    div.className = 'ann-item';
    div.innerHTML =
      '<button class="delete-btn" onclick="deleteAnnotation(' + i + ')">x</button>' +
      '<div class="ann-coords">(' + a.coordX.toFixed(1) + ', ' + a.coordY.toFixed(1) + ', ' + a.coordZ.toFixed(1) + ') mm</div>' +
      '<div class="ann-size">' + a.diameter_mm.toFixed(1) + ' mm diameter</div>' +
      (a.note ? '<div style="color:#888;font-size:11px;margin-top:2px;">' + a.note + '</div>' : '');
    // Click annotation to jump to that slice
    div.addEventListener('click', (e) => {
      if (e.target.classList.contains('delete-btn')) return;
      if (!volumeInfo) return;
      const sliceIdx = Math.round((a.coordZ - volumeInfo.origin[2]) / volumeInfo.spacing[2]);
      loadSlice(Math.max(0, Math.min(volumeInfo.num_slices - 1, sliceIdx)));
    });
    list.appendChild(div);
  }
}

// ---- Actions ----
async function importScan() {
  const path = document.getElementById('import-path').value.trim();
  const label = document.getElementById('import-label').value.trim();
  if (!path) { alert('Enter a DICOM folder path'); return; }
  setStatus('Importing...');
  try {
    const scan = await api('POST', '/api/import', { path, label });
    document.getElementById('import-path').value = '';
    document.getElementById('import-label').value = '';
    await loadScans();
    selectScan(scan.series_uid);
    setStatus('Import complete: ' + (scan.label || scan.series_uid));
  } catch (e) { alert('Import failed: ' + e.message); setStatus('Import failed'); }
}

async function markNegative() {
  if (!currentScan) return;
  if (!confirm('Mark this scan as having no nodules? This will clear existing annotations.')) return;
  await api('POST', '/api/scans/' + currentScan.series_uid + '/mark-negative');
  annotations = [];
  drawFrame();
  renderAnnotations();
  loadScans();
  setStatus('Scan marked as negative');
}

async function prepareData() {
  setStatus('Preparing training data...');
  try {
    const result = await api('POST', '/api/prepare');
    alert('Training data prepared at: ' + result.dataset_dir);
    setStatus('Training data ready at ' + result.dataset_dir);
  } catch (e) { alert('Prepare failed: ' + e.message); setStatus('Prepare failed'); }
}

function changeWindow() {
  // Window change requires re-fetching slices with different params
  // For now we just reload current slice (server uses fixed lung window)
  if (currentScan) loadSlice(currentSlice);
}

function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

// ---- Init ----
loadScans();
</script>
</body>
</html>
"""
