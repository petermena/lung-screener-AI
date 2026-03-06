"""Lightweight ONNX-based inference engine for offline deployment.

This module replaces the PyTorch-based NoduleDetector with an ONNX Runtime
backend. This allows deployment on machines without PyTorch installed,
reducing the install footprint from ~2GB to ~50MB.

GPU execution is automatic when supported hardware and the matching
onnxruntime package are present:

  Provider priority (highest to lowest):
    1. TensorRTExecutionProvider   — NVIDIA GPU via TensorRT (fastest)
    2. CUDAExecutionProvider       — NVIDIA GPU via CUDA
    3. DmlExecutionProvider        — DirectML: Windows AMD / Intel / NVIDIA
    4. ROCMExecutionProvider       — AMD GPU on Linux
    5. CPUExecutionProvider        — always available as fallback

  Required packages per provider:
    CUDA / TensorRT  →  pip install onnxruntime-gpu
    DirectML         →  pip install onnxruntime-directml
    ROCm             →  pip install onnxruntime-rocm
    CPU only         →  pip install onnxruntime

Used by the offline packaged distribution.
"""

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .inference import NoduleFinding, ScanResult, compute_image_number, estimate_lobe
from .preprocessing import CTPreprocessor, extract_patch

logger = logging.getLogger(__name__)


def _build_providers(device_id: int = 0, trt_cache_dir: str | None = None) -> list:
    """Return an ordered provider list for onnxruntime.InferenceSession.

    Probes which providers are actually available in the installed
    onnxruntime package and builds the list from highest- to
    lowest-performance option.  Each provider entry is either a plain
    string (for providers that need no options) or a (name, options)
    tuple.

    Args:
        device_id:      GPU device index (0 = first GPU).
        trt_cache_dir:  Directory for TensorRT engine cache files.
                        Speeds up repeated runs on the same model.
                        Defaults to a sub-directory next to the model.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers: list = []

    # ── 1. TensorRT (best NVIDIA throughput) ─────────────────────────────────
    if "TensorrtExecutionProvider" in available:
        trt_opts: dict = {
            "device_id": device_id,
            "trt_max_workspace_size": 2 * 1024 ** 3,   # 2 GB workspace
            "trt_fp16_enable": True,                    # FP16 for speed
            "trt_engine_cache_enable": True,
        }
        if trt_cache_dir:
            trt_opts["trt_engine_cache_path"] = trt_cache_dir
        providers.append(("TensorrtExecutionProvider", trt_opts))

    # ── 2. CUDA ───────────────────────────────────────────────────────────────
    if "CUDAExecutionProvider" in available:
        cuda_opts: dict = {
            "device_id": device_id,
            # Let the arena grow in power-of-two steps to reduce fragmentation
            "arena_extend_strategy": "kNextPowerOfTwo",
            # Cap at 6 GB so the host OS stays responsive on a workstation
            "gpu_mem_limit": 6 * 1024 ** 3,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        providers.append(("CUDAExecutionProvider", cuda_opts))

    # ── 3. DirectML (Windows — AMD / Intel / NVIDIA) ─────────────────────────
    if "DmlExecutionProvider" in available:
        providers.append(("DmlExecutionProvider", {"device_id": device_id}))

    # ── 4. ROCm (Linux AMD GPU) ───────────────────────────────────────────────
    if "ROCMExecutionProvider" in available:
        providers.append(("ROCMExecutionProvider", {"device_id": device_id}))

    # ── 5. CPU fallback ───────────────────────────────────────────────────────
    providers.append("CPUExecutionProvider")

    return providers


def _probe_max_batch(session) -> int:
    """Detect whether the ONNX model supports batch sizes > 1.

    Some models are exported with a fixed batch dimension of 1 (no
    dynamic_axes).  We detect this by inspecting the first input's
    shape: if the batch axis is a concrete integer (not None / symbolic),
    we return that value; otherwise we return a large sentinel (64) to
    allow normal batching.
    """
    try:
        inp = session.get_inputs()[0]
        batch_dim = inp.shape[0] if inp.shape else None
        if isinstance(batch_dim, int) and batch_dim > 0:
            return batch_dim
    except Exception:
        pass
    return 64  # dynamic batch — use the configured batch_size


class NoduleDetectorONNX:
    """ONNX Runtime-based nodule detector for offline/lightweight deployment."""

    def __init__(
        self,
        config: dict,
        onnx_path: str | Path,
        device_id: int = 0,
    ):
        import onnxruntime as ort

        self.config = config
        self.preprocessor = CTPreprocessor(config)

        inf_config = config.get("inference", {})
        self.threshold = inf_config.get("threshold", 0.15)
        self.nms_distance_mm = inf_config.get("nms_distance_mm", 10.0)
        self.batch_size = inf_config.get("batch_size", 64)

        onnx_path = Path(onnx_path)

        # TensorRT engine cache alongside the model file
        trt_cache = str(onnx_path.parent / "trt_cache")

        providers = _build_providers(device_id=device_id, trt_cache_dir=trt_cache)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
        )

        active = self.session.get_providers()
        active_str = active[0] if active else "unknown"

        # Map provider name to a human-readable description
        _labels = {
            "TensorrtExecutionProvider": "NVIDIA GPU (TensorRT)",
            "CUDAExecutionProvider":     "NVIDIA GPU (CUDA)",
            "DmlExecutionProvider":      "GPU (DirectML)",
            "ROCMExecutionProvider":     "AMD GPU (ROCm)",
            "CPUExecutionProvider":      "CPU",
        }
        label = _labels.get(active_str, active_str)
        logger.info("ONNX Runtime inference device: %s", label)

        if active_str == "CPUExecutionProvider" and len(providers) > 1:
            logger.warning(
                "GPU provider requested but not available — running on CPU. "
                "For CUDA support install: pip install onnxruntime-gpu"
            )

        # Detect the maximum batch size the model supports.
        # Models exported without dynamic_axes have a fixed batch = 1;
        # trying to feed a larger batch triggers an ONNX Reshape error.
        model_max_batch = _probe_max_batch(self.session)
        if model_max_batch < self.batch_size:
            logger.info(
                "ONNX model has fixed batch size %d — "
                "candidates will be classified one at a time",
                model_max_batch,
            )
            self.batch_size = model_max_batch

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

            # Run inference in batches.
            # self.batch_size is already capped to the model's fixed batch size
            # (1 for models exported without dynamic_axes), so iterating with
            # step=self.batch_size processes exactly what the model accepts.
            all_probs = []
            for i in range(0, len(patches_array), self.batch_size):
                batch = patches_array[i : i + self.batch_size]
                outputs = self.session.run(None, {"input": batch})
                logits = outputs[0]  # (B, 2)

                # Numerically-stable softmax
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                all_probs.extend(probs[:, 1].tolist())

            # Build findings
            findings = []
            for cand, prob in zip(candidates, all_probs):
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
