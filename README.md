# Lung Screener AI

AI-powered lung nodule detection for CT scans with GE Centricity PACS integration.

## Overview

Lung Screener AI is a deep learning system that detects and classifies lung nodules on chest CT scans. It provides:

- **3D CNN models** (ResNet3D / DenseNet3D) trained on the LUNA16 dataset
- **Automated Lung-RADS categorization** based on nodule size
- **DICOM networking** for direct PACS integration via C-STORE SCP/SCU
- **Structured Report generation** to send findings back to PACS

> **Disclaimer**: This is a research prototype. Clinical use requires FDA 510(k) clearance. All AI findings must be reviewed by a qualified radiologist.

## Architecture

```
CT Scan (DICOM) → Preprocessing → Candidate Detection → 3D CNN Classification → Findings
                                                                                    ↓
                                                                           DICOM SR → PACS
```

**Pipeline stages:**
1. Load DICOM series and resample to isotropic 1mm spacing
2. Apply HU windowing (-1200 to 600) and normalize
3. Segment lung parenchyma via thresholding + morphology
4. Extract nodule candidates using connected component analysis
5. Classify each candidate with a 3D ResNet/DenseNet
6. Apply non-maximum suppression
7. Generate DICOM Structured Report with Lung-RADS categories

## Setup

```bash
# Clone and install
git clone <repo-url> && cd lung-screener-AI
pip install -e ".[dev]"

# Download LUNA16 dataset (required for training)
# https://luna16.grand-challenge.org/Download/
# Place in ./data/luna16/
```

## Usage

### Train a model

```bash
# Train with default config
lung-screener train --checkpoint-dir ./checkpoints

# Custom config and hyperparameters
lung-screener -c config/custom.yaml train --epochs 50 --batch-size 16 --lr 0.0005

# Resume from checkpoint
lung-screener train --resume ./checkpoints/latest.pth
```

### Run inference on a scan

```bash
# On a DICOM directory
lung-screener predict ./path/to/dicom/series -m ./checkpoints/best.pth

# On a LUNA16 .mhd file, output as JSON
lung-screener predict ./data/luna16/subset0/1.3.6.1.4.1.14519.mhd -m best.pth -o results.json --format json
```

### PACS integration

```bash
# Verify PACS connectivity
lung-screener verify

# Start DICOM listener (receives studies, runs detection, sends SR back)
lung-screener serve -m ./checkpoints/best.pth --port 11112
```

**GE Centricity PACS configuration:**
1. Add this system as a DICOM destination in Centricity:
   - AE Title: `LUNG_SCREEN_AI`
   - Host: `<server-ip>`
   - Port: `11112`
2. Configure auto-routing rules to push chest CT studies to this destination
3. Edit `config/default.yaml` to set your Centricity connection details under `pacs:`

## Configuration

All settings are in `config/default.yaml`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `model.architecture` | `resnet3d` | `resnet3d` or `densenet3d` |
| `model.patch_size` | `[48,48,48]` | 3D patch size for candidates |
| `preprocessing.target_spacing` | `[1,1,1]` | Isotropic resampling (mm) |
| `inference.threshold` | `0.5` | Detection confidence threshold |
| `pacs.local_port` | `11112` | DICOM listener port |
| `pacs.remote_ae_title` | `GEPACS` | Your Centricity AE title |

## Project Structure

```
lung_screener/
├── __init__.py
├── cli.py              # Command-line interface
├── model.py            # 3D CNN architectures (ResNet3D, DenseNet3D)
├── preprocessing.py    # DICOM loading, resampling, lung segmentation
├── dataset.py          # LUNA16 data loading and augmentation
├── train.py            # Training loop with mixed precision
├── inference.py        # End-to-end detection pipeline
└── pacs.py             # DICOM networking and SR generation
```

## Training Data

This system is designed to train on the [LUNA16](https://luna16.grand-challenge.org/) dataset, which is derived from the LIDC-IDRI collection. The dataset contains:

- 888 CT scans with expert nodule annotations
- Annotations include nodule location (x, y, z) and diameter
- Candidate locations with class labels (nodule / non-nodule)

## Lung-RADS Categories

The system assigns [ACR Lung-RADS v2022](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads) categories with type-specific thresholds for solid, part-solid, and ground-glass nodules.

### Solid Nodules

| Category | Size | Recommendation |
|----------|------|----------------|
| 1 | No nodules | Continue annual screening with LDCT in 12 months |
| 2 | <6mm | Continue annual screening with LDCT in 12 months |
| 3 | 6–8mm | Short-term follow-up — LDCT in 6 months |
| 4A | 8–15mm | LDCT in 3 months, PET/CT may be considered |
| 4B | ≥15mm | Tissue sampling and/or PET/CT; multidisciplinary consultation |

### Part-Solid (Subsolid) Nodules

| Category | Size | Recommendation |
|----------|------|----------------|
| 2 | <6mm total | Continue annual screening with LDCT in 12 months |
| 3 | ≥6mm total, solid component <6mm | Short-term follow-up — LDCT in 6 months |
| 4A | Solid component 6–8mm | LDCT in 3 months, PET/CT may be considered |
| 4B | Solid component ≥8mm | Tissue sampling and/or PET/CT; multidisciplinary consultation |

### Ground-Glass Nodules (GGN)

| Category | Size | Recommendation |
|----------|------|----------------|
| 2 | <30mm | Continue annual screening with LDCT in 12 months |
| 3 | ≥30mm | Short-term follow-up — LDCT in 6 months |

> **Risk-based upgrade:** When a Brock/PanCan malignancy probability ≥15% is computed, a Category 3 nodule is automatically upgraded to 4A regardless of type.
