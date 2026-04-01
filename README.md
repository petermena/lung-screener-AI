# lung-screener-AI

End-to-end starter project for **lung-cancer CT screening** with:
1. Transfer learning from **publicly available pretrained models** (MedicalNet / MONAI backbones).
2. A lightweight **PACS-style viewer UI** (Streamlit) for uploading and reviewing your own scans.

> ⚠️ This project is for research/prototyping only and is **not** a medical device.

---

## Why this fits your EC2 setup (g4dn.xlarge + 300GB)

- `g4dn.xlarge` gives 1x NVIDIA T4 GPU (16GB VRAM), which is sufficient for 3D transfer learning with small batch sizes.
- Storage-heavy CT workflows are addressed by:
  - keeping raw scans in compressed archives,
  - using a CSV index for scans,
  - and only loading batches on demand.
- Defaults are tuned conservatively (`batch_size=2`, resized volume `96x192x192`) for memory stability.

---

## Project layout

```bash
lung-screener-AI/
├── app.py                    # Streamlit PACS-style uploader/viewer + inference
├── requirements.txt
└── lung_screener/
    ├── data.py               # DICOM/NIfTI loading + preprocessing
    ├── inference.py          # Programmatic prediction utility
    ├── model.py              # Pretrained model factory
    └── train.py              # Fine-tuning script
```

---

## 1) Environment setup on EC2

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(Optional) verify GPU:

```bash
python - <<'PY'
import torch
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

---

## 2) Training data format

Create a CSV file with columns:
- `scan_path`: path to either
  - a `.nii/.nii.gz/.mha/.mhd` volume, or
  - a directory containing `.dcm` slices.
- `label`: binary class (`0=negative`, `1=suspicious`)

Example (`data/train.csv`):

```csv
scan_path,label
/data/lidc/case_0001,0
/data/lidc/case_0002,1
/data/other/case_0100.nii.gz,1
```

---

## 3) Train with publicly pretrained backbones

### Option A (default): MedicalNet ResNet-18 (3D)
- Weights are downloaded from public MedicalNet release on first run.

```bash
python -m lung_screener.train \
  --csv data/train.csv \
  --output checkpoints \
  --backbone medicalnet_resnet18 \
  --epochs 20 \
  --batch-size 2 \
  --lr 1e-4
```

If your dataset is tiny (for example only 1 sample in a class), use:

```bash
python -m lung_screener.train \
  --csv data/train.csv \
  --output checkpoints \
  --backbone medicalnet_resnet18 \
  --val-size 0 \
  --no-stratify
```

### Option B: MONAI DenseNet121 (3D)

```bash
python -m lung_screener.train \
  --csv data/train.csv \
  --output checkpoints \
  --backbone monai_densenet121
```

Best checkpoint is saved as `checkpoints/best.pt`.

---

## 4) Run the PACS-like UI

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then open your EC2 public IP on port `8501`.

### UI features
- Upload single scan volume (`.nii/.mha`) or zipped DICOM series (`.zip`).
- Scroll through axial slices.
- Quick multi-slice preview.
- Run inference with checkpoint + pretrained backbone.

---

## 5) Inference from Python

```python
from pathlib import Path
from lung_screener.inference import predict_scan

result = predict_scan(
    scan_path=Path('/data/patient123'),
    checkpoint_path=Path('checkpoints/best.pt'),
    backbone='medicalnet_resnet18',
    device='cuda',
)
print(result)
```

---

## Operational notes for your 300GB budget

- Keep raw datasets in `/data/raw` and preprocessed cache in `/data/cache`.
- Use gzip-compressed NIfTI where possible.
- Remove temporary upload artifacts frequently (`/tmp`, Streamlit upload temp folders).
- Persist checkpoints to EBS and snapshot regularly.

---

## Next high-impact improvements

1. Add candidate nodule detection (2-stage: detector + malignancy classifier).
2. Add calibration + threshold tuning on validation cohort.
3. Add DICOM metadata panel (Series UID, spacing, kernel, dose).
4. Add docker-compose for one-command deploy on EC2.
5. Add audit logging + auth before exposing publicly.


---

## Troubleshooting

- **Error: least populated class has only 1 member**
  - Add more samples per class for stratified splitting, or pass `--no-stratify`.
  - For very small experiments, disable validation split with `--val-size 0`.
