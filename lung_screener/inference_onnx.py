"""Lightweight ONNX-based inference engine for offline deployment.

This module replaces the PyTorch-based NoduleDetector with an ONNX Runtime
backend. This allows deployment on machines without PyTorch installed,
reducing the install footprint from ~2GB to ~50MB.

Used by the offline packaged distribution.
"""

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .inference import NoduleFinding, ScanResult, compute_image_number, estimate_lobe
from .preprocessing import CTPreprocessor, extract_patch

logger = logging.getLogger(__name__)


class NoduleDetectorONNX:
    """ONNX Runtime-based nodule detector for offline/lightweight deployment."""

    def __init__(
        self,
        config: dict,
        onnx_path: str | Path,
    ):
        import onnxruntime as ort

        self.config = config
        self.preprocessor = CTPreprocessor(config)

        inf_config = config.get("inference", {})
        self.threshold = inf_config.get("threshold", 0.5)
        self.nms_distance_mm = inf_config.get("nms_distance_mm", 10.0)
        self.batch_size = inf_config.get("batch_size", 64)

        # Create ONNX session — prefer GPU if available, fall back to CPU
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self.session.get_providers()
        logger.info(f"ONNX Runtime using: {active[0]}")

    def predict_scan(self, image: sitk.Image, series_uid: str = "") -> ScanResult:
        """Run detection pipeline using ONNX model."""
        try:
            processed = self.preprocessor.process_scan(image)
            volume = processed["volume"]
            candidates = processed["candidates"]
            spacing = processed["spacing"]
            origin = processed["origin"]

            logger.info(f"Found {len(candidates)} candidates in scan")

            # Compute z extent for lobe estimation
            vol_shape = volume.shape
            z_min = origin[2]
            z_max = origin[2] + vol_shape[0] * spacing[2]

            if not candidates:
                return ScanResult(series_uid=series_uid)

            patch_size = tuple(
                self.config.get("model", {}).get("patch_size", [48, 48, 48])
            )

            # Extract patches
            patches = []
            for cand in candidates:
                patch = extract_patch(volume, cand["center_voxel"], patch_size)
                patches.append(patch)

            patches_array = np.stack(patches)[:, np.newaxis, ...].astype(np.float32)

            # Run inference in batches
            all_probs = []
            for i in range(0, len(patches_array), self.batch_size):
                batch = patches_array[i : i + self.batch_size]
                outputs = self.session.run(None, {"input": batch})
                logits = outputs[0]  # (B, 2)

                # Softmax
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                all_probs.extend(probs[:, 1])

            # Build findings
            findings = []
            for idx, (cand, prob) in enumerate(zip(candidates, all_probs)):
                if prob >= self.threshold:
                    center_world = tuple(
                        o + c * s
                        for o, c, s in zip(origin, cand["center_voxel"], spacing)
                    )
                    wx = center_world[2]
                    wy = center_world[1]
                    wz = center_world[0]

                    findings.append(NoduleFinding(
                        x=wx,
                        y=wy,
                        z=wz,
                        diameter_mm=cand["diameter_mm"],
                        confidence=float(prob),
                        lobe=estimate_lobe(wx, wy, wz, z_min, z_max),
                        image_number=compute_image_number(wz, origin[2], spacing[2]),
                        series_uid=series_uid,
                    ))

            findings = self._nms(findings)
            logger.info(f"Detected {len(findings)} nodules after NMS")
            return ScanResult(series_uid=series_uid, findings=findings)

        except Exception as e:
            logger.error(f"Error processing scan: {e}")
            return ScanResult(
                series_uid=series_uid,
                processing_status="error",
                error_message=str(e),
            )

    def _nms(self, findings: list[NoduleFinding]) -> list[NoduleFinding]:
        """Non-maximum suppression based on distance."""
        if len(findings) <= 1:
            return findings

        findings = sorted(findings, key=lambda f: f.confidence, reverse=True)
        keep = []

        for finding in findings:
            too_close = False
            for kept in keep:
                dist = np.sqrt(
                    (finding.x - kept.x) ** 2
                    + (finding.y - kept.y) ** 2
                    + (finding.z - kept.z) ** 2
                )
                if dist < self.nms_distance_mm:
                    too_close = True
                    break
            if not too_close:
                keep.append(finding)

        return keep
