from __future__ import annotations

import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import streamlit as st

from lung_screener.data import load_volume, middle_slices, preprocess_volume
from lung_screener.model import ModelConfig, build_model

import torch

st.set_page_config(page_title="Lung Screener AI", layout="wide")
st.title("Lung Cancer CT Screener + Lightweight PACS")

with st.sidebar:
    st.header("Model")
    backbone = st.selectbox("Pretrained backbone", ["medicalnet_resnet18", "monai_densenet121"])
    checkpoint = st.text_input("Checkpoint path", value="checkpoints/best.pt")
    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Inference device: {run_device}")

uploaded = st.file_uploader(
    "Upload a .nii/.mha file OR a .zip of DICOM slices",
    type=["nii", "gz", "mha", "mhd", "zip"],
)

if uploaded is None:
    st.info("Upload a CT volume to begin.")
    st.stop()

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    if uploaded.name.endswith(".zip"):
        archive_path = tmp / uploaded.name
        archive_path.write_bytes(uploaded.read())
        dicom_dir = tmp / "dicoms"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive_path, "r") as zf:
            zf.extractall(dicom_dir)
        scan_path = dicom_dir
    else:
        scan_path = tmp / uploaded.name
        scan_path.write_bytes(uploaded.read())

    volume = load_volume(scan_path)
    st.success(f"Loaded volume with shape: {volume.shape}")

    st.subheader("Viewer")
    sidx = st.slider("Slice", 0, int(volume.shape[0]) - 1, int(volume.shape[0] // 2))
    slice_img = volume[sidx]
    lo, hi = np.percentile(slice_img, 1), np.percentile(slice_img, 99)
    norm = np.clip((slice_img - lo) / (hi - lo + 1e-6), 0, 1)
    st.image(norm, clamp=True, caption=f"Axial slice {sidx}")

    cols = st.columns(3)
    for col, img in zip(cols, list(middle_slices(volume, count=3))):
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        col.image(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), clamp=True)

    if st.button("Run screening model"):
        model = build_model(ModelConfig(backbone=backbone))
        if Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
        else:
            st.warning("Checkpoint not found. Running with pretrained backbone + random classifier head.")

        x = preprocess_volume(volume).unsqueeze(0).to(run_device)
        model = model.to(run_device).eval()
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

        st.subheader("Screening output")
        st.metric("Suspicious probability", f"{probs[1] * 100:.2f}%")
        st.metric("Negative probability", f"{probs[0] * 100:.2f}%")
        st.caption("Research-use only. Not for clinical diagnosis.")
