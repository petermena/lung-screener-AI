# Lung Cancer CT Screening Module + Lightweight PACS UI

This repo now provides a practical, EC2-friendly baseline for **CT lung screening triage** with:

- a **FastAPI backend** for DICOM ingest / rendering / inference,
- a **Streamlit UI** for upload + interactive slice viewing,
- optional **Orthanc** via Docker Compose for PACS-like workflows.

---

## 1) Recommended pretrained open-source model strategy

For your `g4dn.xlarge` (NVIDIA T4, 16GB VRAM), the most practical path is to run a MONAI bundle-backed workflow and keep the app resilient if model downloads are unavailable.

Current implementation in `app/model.py`:

- targets MONAI bundle name: `lung_nodule_ct_detection` (configurable via env),
- attempts bundle download to `data/model_bundles`,
- gracefully falls back to baseline scoring if MONAI bundle assets are unavailable.

> Important: this is an engineering starter for triage/testing workflows, **not** a clinical diagnostic device.

---

## 2) Features implemented

### API endpoints

- `POST /studies/upload` – upload `.dcm` files or a `.zip` containing DICOM files.
- `GET /studies` – list uploaded studies.
- `GET /studies/{id}/metadata` – return study metadata.
- `GET /studies/{id}/slice/{idx}?center=-600&width=1500` – PNG rendering with windowing.
- `POST /studies/{id}/infer` – run inference adapter + return risk output.

### Lightweight PACS-style UI

- Upload study files directly from browser.
- Choose study from existing list.
- Scroll slices and adjust **window center / width**.
- Launch inference and inspect structured output.

### Optional PACS service

- `docker-compose.yml` includes **Orthanc** for DICOM store/viewer integration.

---

## 3) Local run (Ubuntu DL AMI)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Run backend:

```bash
./scripts/run_api.sh
```

Run UI:

```bash
./scripts/run_ui.sh
```

Open:

- API docs: `http://<ec2-ip>:8000/docs`
- UI: `http://<ec2-ip>:8501`

---

## 4) Docker / Compose run

```bash
docker compose up --build
```

Then access:

- API: `http://<ec2-ip>:8000/docs`
- UI: `http://<ec2-ip>:8501`
- Orthanc UI: `http://<ec2-ip>:8042`

---

## 5) Environment variables

- `LUNG_SCREENER_DATA_ROOT` (default: `data/studies`)
- `LUNG_SCREENER_MODEL_NAME` (default: `lung_nodule_ct_detection`)
- `LUNG_SCREENER_MODEL_BUNDLE_DIR` (default: `data/model_bundles`)
- `LUNG_SCREENER_MODEL_AUTO_DOWNLOAD` (default: `true`)
- `LUNG_SCREENER_DEVICE` (default: `cuda`)

---

## 6) High-impact next steps

1. Wire full MONAI bundle infer workflow with deterministic pre/post transforms.
2. Add 3D MPR (axial/coronal/sagittal) and nodule overlays.
3. Add structured report export (JSON + PDF).
4. Add auth and audit logging for multi-user use.
5. Add DICOMweb bridge for production PACS interoperability.
