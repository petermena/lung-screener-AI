# Lung Screener AI

AI-powered lung nodule detection for chest CT scans — with an integrated web viewer, PACS integration, and fully offline operation.

> **Disclaimer:** Research prototype. Clinical use requires FDA 510(k) clearance. All AI findings must be reviewed by a qualified radiologist before clinical action.

---

## What it does

- Detects and measures lung nodules in chest CT scans with **AUC-ROC 0.987** (5-fold CV)
- Assigns **ACR Lung-RADS v2022** categories with type-specific thresholds (solid / part-solid / ground-glass)
- Generates **dictation-ready radiology reports** — separate from PowerScribe, copy-paste ready
- Provides an **interactive browser-based DICOM viewer** with windowing, zoom, pan, and measurement tools, with nodule locations marked directly on the images
- Accepts studies via **drag-and-drop upload**, **DICOM C-STORE** from PACS, or **command-line batch** processing
- Runs **completely offline** — no internet connection needed after installation

---

## Quick start

```bash
git clone <repo-url> && cd lung-screener-AI
pip install -e ".[server]"

# Start the viewer with your trained model
lung-screener viewer -m checkpoints/best.pth

# Open http://localhost:8080 — drag DICOM files into the browser
```

---

## Web Viewer

The viewer is a browser-based DICOM workstation. Start it with one command:

```bash
lung-screener viewer -m checkpoints/best.pth
lung-screener viewer -m checkpoints/best.pth --port 9090   # custom port
lung-screener viewer -m checkpoints/fold_3/best.pth        # specific fold
```

Then open **http://localhost:8080** in any browser (Chrome, Firefox, Edge).

### How to use

1. **Upload a study** — drag and drop `.dcm` files or a `.zip` archive onto the left panel, or click to browse
2. **Wait for analysis** — the model runs automatically; a status indicator shows progress
3. **Review findings** — nodules appear as color-coded circles directly on the images; click any finding card to jump to that slice
4. **Adjust the image** — use windowing presets (Lung / Mediastinum / Bone), or drag the WW/WC sliders
5. **Measure** — select the Measure tool and draw a line across any structure
6. **Copy the report** — the right panel shows a full structured report; click Copy to paste it anywhere

### Viewer tools

| Tool | How to activate | Action |
|------|----------------|--------|
| Window / Level | `W/L` button or left-click drag on image | Adjust contrast |
| Pan | Middle-click drag | Move image |
| Zoom | `Zoom` button + left-click drag | Zoom in/out |
| Measure | `Measure` button + click-drag | Draw a calibrated ruler (mm) |
| Scroll slices | Mouse wheel | Navigate through CT slices |
| Arrow keys | ↑ ↓ → ← | Previous / next slice |

### Window presets

| Preset | WW / WC | Best for |
|--------|---------|----------|
| Lung | 1500 / −600 | Lung parenchyma, nodule shape |
| Mediastinum | 350 / 40 | Soft tissue, lymph nodes |
| Bone | 2000 / 300 | Ribs, vertebrae |

### Nodule overlay colors

| Color | Lung-RADS | Recommendation |
|-------|-----------|----------------|
| 🟢 Green | 1–2 | Routine annual screening |
| 🟡 Yellow | 3 | 6-month follow-up LDCT |
| 🟠 Orange | 4A | 3-month follow-up or PET/CT |
| 🔴 Red | 4B / 4X | Tissue sampling / multidisciplinary review |

### Report

The report panel on the right shows a complete structured radiology report generated automatically from the AI findings. It is **independent of PowerScribe** — use the Copy button to paste it into any reporting system, EHR, or document. The report follows standard radiology prose style and includes:

- Findings section (lobe location, size, type, Lung-RADS category, calcification)
- Impression (overall Lung-RADS with clinical language)
- Recommendation (ACR-aligned follow-up interval)

---

## Architecture

```
CT scan (DICOM) ──► Preprocessing ──► Lung segmentation ──► Candidate extraction
                                                                      │
                                                              3D patches (48³)
                                                                      │
                                                          SE-ResNeXt3D classifier
                                                                      │
                                               ┌──────────────────────┤
                                               ▼                      ▼
                                        NMS + filtering       Calcification analysis
                                               │
                                  Lung-RADS v2022 categorization
                                  (type-specific: solid/part-solid/GGN)
                                               │
                               ┌───────────────┼───────────────┐
                               ▼               ▼               ▼
                          Web viewer     Text report      DICOM SR → PACS
```

**Model:** SE-ResNeXt3D — grouped convolutions (cardinality 32), squeeze-excitation attention, multi-scale fusion, stochastic depth, Stochastic Weight Averaging (SWA)

**Training:** 5-fold cross-validation on LUNA16 + LUNA25, focal loss, test-time augmentation (8 flips/rotations), cosine warm restarts

---

## Installation

### Standard

```bash
pip install -e .            # core (training, inference, PACS)
pip install -e ".[server]"  # + web viewer
pip install -e ".[dev]"     # + testing tools
```

### Fully offline (air-gapped machine)

The viewer and all JavaScript dependencies are bundled — no internet is needed at runtime. To deploy on a machine that has never had internet access:

```bash
# On a machine WITH internet (one time):
python scripts/package_offline.py \
  --checkpoint checkpoints/best.pth \
  --output dist/ \
  --full          # include PyTorch for GPU; omit for lightweight ONNX mode

# Creates dist/lung-screener-offline.zip
# Copy the zip to the target machine, then:

unzip lung-screener-offline.zip
cd lung-screener-offline
./install.sh            # Linux/macOS
install.bat             # Windows

# Start the viewer — fully offline:
source venv/bin/activate
lung-screener viewer -m model/model.pth
```

**Package sizes:**

| Mode | Approximate size | GPU support |
|------|-----------------|-------------|
| ONNX (default) | ~200 MB | CPU only |
| Full (`--full`) | ~2.5 GB | CPU + CUDA GPU |

---

## Training

### Single-fold

```bash
lung-screener -c config/best_model.yaml train \
  --checkpoint-dir ./checkpoints \
  --dataset luna16 ./data/luna16 \
  --dataset luna25 ./data/luna25
```

### 5-fold cross-validation (recommended)

```bash
lung-screener -c config/best_model.yaml train-kfold \
  -k 5 \
  --checkpoint-dir checkpoints \
  --resume          # safe to re-run; picks up from last completed epoch
```

Checkpoints for each fold are saved under `checkpoints/fold_N/`. Resume is automatic — re-running the same command after an interruption continues from where it left off.

### Resume from checkpoint

```bash
lung-screener train --resume ./checkpoints/latest.pth
```

---

## Inference

### Web viewer (recommended)

```bash
lung-screener viewer -m checkpoints/best.pth
# Open http://localhost:8080 and drag in DICOM files
```

### Command line — single scan

```bash
lung-screener predict ./path/to/dicom/series/ -m checkpoints/best.pth
lung-screener predict ./scan.mhd             -m checkpoints/best.pth -o results.json
```

### Command line — batch processing

```bash
lung-screener batch ./dicom_studies/ -m checkpoints/best.pth -o results/
```

### Ensemble (multiple folds)

```bash
lung-screener ensemble-predict ./path/to/dicom/ \
  --models checkpoints/fold_0/best.pth \
           checkpoints/fold_1/best.pth \
           checkpoints/fold_2/best.pth \
           checkpoints/fold_3/best.pth
```

---

## PACS integration

The system speaks native DICOM networking and can receive studies directly from your PACS.

```bash
# Start DICOM C-STORE listener
lung-screener serve -m checkpoints/best.pth --port 11112

# Verify connectivity to PACS
lung-screener verify
```

**GE Centricity configuration:**
1. Add this system as a DICOM destination in Centricity:
   - AE Title: `LUNG_SCREEN_AI`
   - Host: `<this server's IP>`
   - Port: `11112`
2. Configure auto-routing to push chest CT series to this destination
3. Edit `config/default.yaml` → `pacs:` section with your Centricity host/port/AE title

Results are sent back as a DICOM Structured Report (SR) to the originating PACS.

---

## Model performance

Trained on LUNA16 (888 scans) + LUNA25 (~4,000 scans). Evaluated with 5-fold cross-validation.

### 5-fold cross-validation results

| Fold | AUC-ROC | Sensitivity | Specificity | F1 | ECE | Best Epoch |
|------|---------|-------------|-------------|-----|-----|------------|
| Fold 0 | 0.9865 | 100.0% | 80.6% | 0.485 | 0.062 | 149 |
| Fold 1 | 0.9870 | 99.1% | 87.8% | 0.597 | 0.035 | 149 |
| Fold 2 | 0.9887 | 99.5% | 87.2% | 0.586 | 0.034 | 143 |
| Fold 3 | **0.9898** | 99.5% | 87.7% | 0.596 | 0.033 | 147 |
| Fold 4 | 0.9847 | 97.4% | 88.7% | 0.608 | 0.029 | 136 |
| **Mean ± Std** | **0.9873 ± 0.0018** | **99.1% ± 0.9%** | **86.4% ± 3.0%** | **0.574 ± 0.045** | **0.039 ± 0.012** | — |

All metrics at threshold 0.15. Full results: `checkpoints/kfold_eval_summary.json`.

### Overall metrics (mean across 5 folds @ threshold 0.15)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.9873** (95% CI: 0.9846–0.9901) |
| Sensitivity | **99.1%** (≤ 4.5 missed nodules per fold) |
| Specificity | 86.4% |
| Precision (PPV) | 40.6% |
| NPV | 99.9% |
| F1 Score | 0.574 |
| ECE (calibration) | 0.039 |

### FROC sensitivity

Sensitivity at fixed false-positive rates per scan (mean ± std across 5 folds):

| FP/scan | 0.0125 | 0.025 | 0.05 | 0.1 | 0.2 | 0.4 |
|---------|--------|-------|------|-----|-----|-----|
| Sensitivity | 75.5% ± 2.1% | 82.7% ± 2.0% | 92.1% ± 1.6% | 97.9% ± 0.8% | 100% | 100% |

### Threshold trade-offs

Mean across 5 folds:

| Threshold | Sensitivity | Specificity | Precision |
|-----------|-------------|-------------|-----------|
| 0.10 | 99.7% | 83.7% | 33.9% |
| **0.15** (detection threshold) | 99.1% | 86.4% | 40.6% |
| 0.20 | 97.9% | 89.7% | 47.2% |
| 0.30 | 94.5% | 93.7% | 58.2% |
| 0.50 | 78.2% | 98.4% | 82.3% |
| 0.90 | 40.8% | 99.9% | 97.8% |

---

## Configuration

Key settings in `config/default.yaml` (override with `config/best_model.yaml` for best performance):

| Setting | Default | Description |
|---------|---------|-------------|
| `model.architecture` | `se_resnext3d` | `resnet3d`, `densenet3d`, or `se_resnext3d` |
| `model.patch_size` | `[48,48,48]` | 3D patch size for candidates (voxels) |
| `model.predict_nodule_type` | `true` | Classify solid / part-solid / ground-glass |
| `preprocessing.target_spacing` | `[1,1,1]` | Isotropic resampling spacing (mm) |
| `inference.threshold` | `0.15` | Detection confidence threshold (Youden's J) |
| `inference.nms_distance_mm` | `10.0` | Non-maximum suppression radius |
| `inference.tta.enabled` | `true` | Test-time augmentation (8 flips/rotations) |
| `training.swa.enabled` | `true` | Stochastic Weight Averaging |
| `pacs.local_port` | `11112` | DICOM listener port |
| `pacs.remote_ae_title` | `GEPACS` | Destination PACS AE title |

---

## Project structure

```
lung_screener/
├── api.py                  # FastAPI backend (upload, inference, DICOM serving)
├── cli.py                  # All CLI commands (train, predict, viewer, serve, …)
├── model.py                # SE-ResNeXt3D / ResNet3D / DenseNet3D architectures
├── inference.py            # End-to-end detection pipeline, NoduleDetector
├── preprocessing.py        # DICOM loading, resampling, lung segmentation
├── dataset.py              # LUNA16/LUNA25 data loading and augmentation
├── train.py                # Training loop (mixed precision, SWA, k-fold)
├── evaluate.py             # AUC, FROC, calibration metrics
├── pacs.py                 # DICOM C-STORE SCP/SCU, Structured Report generation
├── risk_model.py           # Brock/PanCan malignancy scoring, Lung-RADS v2022
├── calcification.py        # Benign calcification pattern detection
├── calibration.py          # Confidence calibration (temperature scaling)
├── fp_reduction.py         # 2nd-stage false positive reduction
├── gradcam.py              # Grad-CAM saliency visualization
├── feedback.py             # Radiologist feedback collection
├── retrain.py              # Incremental retraining from feedback
├── active_learning.py      # Uncertainty sampling
├── prior_comparison.py     # Nodule growth tracking vs. prior studies
├── export.py               # ONNX export
├── optimize.py             # Model quantization / pruning
├── static/
│   ├── viewer.html         # Self-contained browser DICOM viewer
│   └── vendor/             # Bundled JavaScript (Cornerstone.js — no CDN needed)
└── …

config/
├── default.yaml            # Default hyperparameters
├── best_model.yaml         # Optimized config (SE-ResNeXt3D, SWA, TTA, A10G GPU)
└── fp_reduction.yaml       # 2nd-stage classifier config

scripts/
├── package_offline.py      # Bundle everything for air-gapped deployment
├── run_kfold_cv.sh         # Launch k-fold training
├── inspect_fold_checkpoint.py  # Diagnose k-fold checkpoints
├── aggregate_kfold_eval.py     # Aggregate per-fold eval results into summary JSON
└── …
```

---

## Lung-RADS categories

The system assigns [ACR Lung-RADS v2022](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads) categories automatically, with type-specific thresholds.

### Solid nodules

| Category | Diameter | Recommendation |
|----------|----------|----------------|
| 1 | None detected | Annual screening LDCT in 12 months |
| 2 | < 6 mm | Annual screening LDCT in 12 months |
| 3 | 6–8 mm | LDCT in 6 months |
| 4A | 8–15 mm | LDCT in 3 months; PET/CT may be considered |
| 4B | ≥ 15 mm | Tissue sampling and/or PET/CT; multidisciplinary review |

### Part-solid (subsolid) nodules

| Category | Size | Recommendation |
|----------|------|----------------|
| 2 | < 6 mm total | Annual screening LDCT in 12 months |
| 3 | ≥ 6 mm total, solid component < 6 mm | LDCT in 6 months |
| 4A | Solid component 6–8 mm | LDCT in 3 months; PET/CT may be considered |
| 4B | Solid component ≥ 8 mm | Tissue sampling and/or PET/CT |

### Ground-glass nodules (GGN)

| Category | Size | Recommendation |
|----------|------|----------------|
| 2 | < 30 mm | Annual screening LDCT in 12 months |
| 3 | ≥ 30 mm | LDCT in 6 months |

### Category 4X — additional suspicious features

Applied to any Category 3–4 nodule with spiculated margins or interval growth. Recommendation: tissue sampling and/or PET/CT with multidisciplinary consultation.

> **Risk-based upgrade:** Brock/PanCan malignancy probability ≥ 15% automatically upgrades a Category 3 nodule to 4A.

---

## Training data

| Dataset | Scans | Source |
|---------|-------|--------|
| [LUNA16](https://luna16.grand-challenge.org/) | 888 | LIDC-IDRI; expert nodule annotations with location and diameter |
| [LUNA25](https://luna25.grand-challenge.org/) | ~4,000 | Additional annotated nodule blocks and full volumes |
