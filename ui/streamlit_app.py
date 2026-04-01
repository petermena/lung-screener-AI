from __future__ import annotations

import requests
import streamlit as st

API = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("Lung CT Screener + Lightweight PACS Viewer")
st.caption("Upload a CT series (.dcm files or a .zip), browse slices, and run triage inference.")

uploaded = st.file_uploader("Upload DICOM slices", type=["dcm", "zip"], accept_multiple_files=True)
if st.button("Upload Study", use_container_width=True):
    if not uploaded:
        st.warning("Upload at least one .dcm file first.")
    else:
        files = [("files", (f.name, f.getvalue(), "application/dicom")) for f in uploaded]
        r = requests.post(f"{API}/studies/upload", files=files, timeout=120)
        if r.ok:
            st.success(f"Uploaded study {r.json()['study_id']} ({r.json()['saved_files']} slices)")
            st.session_state["study_id"] = r.json()["study_id"]
        else:
            st.error(r.text)

resp = requests.get(f"{API}/studies", timeout=20)
studies = resp.json().get("studies", []) if resp.ok else []
selected = st.selectbox(
    "Select study",
    options=studies if studies else [""],
    index=len(studies) - 1 if studies else 0,
)

study_id = st.session_state.get("study_id") or selected
if study_id:
    meta_r = requests.get(f"{API}/studies/{study_id}/metadata", timeout=30)
    if meta_r.ok:
        meta = meta_r.json()
        st.write("**Study metadata**", meta)
        max_idx = int(meta["num_slices"]) - 1
        idx = st.slider("Slice", min_value=0, max_value=max_idx, value=max_idx // 2)
        col1, col2 = st.columns(2)
        window_center = col1.slider("Window Center", min_value=-1000, max_value=500, value=-600)
        window_width = col2.slider("Window Width", min_value=200, max_value=2500, value=1500)
        st.image(
            f"{API}/studies/{study_id}/slice/{idx}?center={window_center}&width={window_width}",
            caption=f"{study_id} - slice {idx}",
        )

        if st.button("Run Inference", use_container_width=True):
            infer_r = requests.post(f"{API}/studies/{study_id}/infer", timeout=180)
            if infer_r.ok:
                out = infer_r.json()
                st.metric("Risk Score", out["risk_score"])
                st.json(out)
            else:
                st.error(infer_r.text)
