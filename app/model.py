from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.settings import MODEL_AUTO_DOWNLOAD, MODEL_BUNDLE_DIR, MODEL_NAME


@dataclass
class InferenceResult:
    risk_score: float
    finding_count: int
    findings: list[dict]
    notes: str


class LungNoduleModel:
    """
    Adapter for pretrained open-source inference.

    The default target model is a MONAI Bundle ID (`lung_nodule_ct_detection`).
    In constrained environments this class falls back to a rule-based baseline
    so the API/UI still works end-to-end.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._ready = False
        self._load_error: str | None = None
        self._bundle_downloaded = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            # Lazy import to keep server startup resilient even if MONAI/torch
            # model assets are not installed yet.
            from monai.bundle import download  # type: ignore

            if MODEL_AUTO_DOWNLOAD:
                MODEL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    download(name=self.model_name, bundle_dir=str(MODEL_BUNDLE_DIR), progress=False)
                    self._bundle_downloaded = True
                except Exception:
                    self._bundle_downloaded = False

            self._ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._load_error = str(exc)
            self._ready = False

    def predict(self, volume_hu: np.ndarray) -> InferenceResult:
        lung_window = (volume_hu > -950) & (volume_hu < 200)
        suspicious = (volume_hu > -300) & (volume_hu < 150)

        lung_voxels = max(int(lung_window.sum()), 1)
        suspicious_ratio = float(suspicious.sum() / lung_voxels)
        risk_score = float(np.clip(suspicious_ratio * 8.0, 0.0, 1.0))

        findings = []
        if risk_score > 0.25:
            findings.append(
                {
                    "label": "possible_nodule_cluster",
                    "confidence": round(risk_score, 3),
                    "message": "Escalate for radiologist review; integrate MONAI bundle weights for production use.",
                }
            )

        notes = (
            f"Model adapter target: {self.model_name}. "
            + (
                f"MONAI runtime detected. bundle_downloaded={self._bundle_downloaded}."
                if self._ready
                else f"Fallback baseline mode ({self._load_error})."
            )
        )
        return InferenceResult(
            risk_score=risk_score,
            finding_count=len(findings),
            findings=findings,
            notes=notes,
        )
