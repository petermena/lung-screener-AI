"""ONNX model optimization: graph optimization and quantization.

Applies ONNX Runtime graph optimizations (constant folding, node fusion)
and optional INT8 dynamic quantization to reduce model size and improve
inference latency.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def optimize_onnx(
    input_path: str | Path,
    output_path: str | Path,
    quantize: bool = False,
) -> dict:
    """Optimize an ONNX model with graph transformations and optional quantization.

    Args:
        input_path: Path to the source .onnx model.
        output_path: Where to save the optimized .onnx model.
        quantize: If True, apply dynamic INT8 quantization after graph optimization.

    Returns:
        Dict with optimization results (sizes, reduction).
    """
    import onnx
    from onnx import shape_inference
    import onnxruntime as ort

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_size = input_path.stat().st_size

    # --- Step 1: Graph optimization via ORT session options ---
    # Use EXTENDED level to avoid hardware-specific transforms that break portability
    optimized_tmp = output_path.parent / f"{output_path.stem}_graphopt.onnx"
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(optimized_tmp)

    ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])
    logger.info("Graph optimization complete")

    # --- Step 2: Optional INT8 dynamic quantization ---
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Run shape inference on the graph-optimized model first —
        # quantize_dynamic requires complete shape information.
        preproc_path = output_path.parent / f"{output_path.stem}_preproc.onnx"
        model = onnx.load(str(optimized_tmp))
        try:
            inferred = shape_inference.infer_shapes(model, data_prop=True)
            onnx.save(inferred, str(preproc_path))
            quant_input = str(preproc_path)
        except Exception:
            logger.warning("Shape inference failed, quantizing from graph-optimized model directly")
            quant_input = str(optimized_tmp)
            preproc_path = None

        try:
            quantize_dynamic(
                model_input=quant_input,
                model_output=str(output_path),
                weight_type=QuantType.QUInt8,
            )
            logger.info("Dynamic INT8 quantization complete")
        except Exception as e:
            # Fall back to graph-optimized only if quantization fails
            logger.warning("Quantization failed (%s), using graph-optimized model", e)
            quantize = False
            optimized_tmp.rename(output_path)

        # Clean up temp files
        if optimized_tmp.exists():
            optimized_tmp.unlink()
        if preproc_path and preproc_path.exists():
            preproc_path.unlink()
    else:
        optimized_tmp.rename(output_path)

    final_size = output_path.stat().st_size
    reduction_pct = (1 - final_size / original_size) * 100

    # --- Step 3: Validate optimized model runs ---
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    import numpy as np
    dummy = np.random.randn(1, 1, 48, 48, 48).astype(np.float32)
    sess.run(None, {"input": dummy})
    logger.info("Optimized model inference validation passed")

    results = {
        "original_size_mb": round(original_size / 1024 / 1024, 2),
        "optimized_size_mb": round(final_size / 1024 / 1024, 2),
        "reduction_pct": round(reduction_pct, 1),
        "graph_optimized": True,
        "quantized": quantize,
        "output_path": str(output_path),
    }

    logger.info(
        "Optimization: %.1fMB -> %.1fMB (%.1f%% reduction)",
        results["original_size_mb"],
        results["optimized_size_mb"],
        results["reduction_pct"],
    )

    return results


def benchmark_onnx(
    model_path: str | Path,
    n_warmup: int = 5,
    n_runs: int = 50,
    batch_size: int = 1,
    patch_size: tuple[int, ...] = (48, 48, 48),
) -> dict:
    """Benchmark ONNX model inference latency and throughput.

    Args:
        model_path: Path to .onnx model.
        n_warmup: Warmup iterations (excluded from timing).
        n_runs: Number of timed iterations.
        batch_size: Batch size for benchmarking.
        patch_size: Input patch dimensions.

    Returns:
        Dict with latency statistics and throughput.
    """
    import time

    import numpy as np
    import onnxruntime as ort

    model_path = Path(model_path)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    provider = sess.get_providers()[0]

    dummy = np.random.randn(batch_size, 1, *patch_size).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {"input": dummy})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(None, {"input": dummy})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies = np.array(latencies)

    results = {
        "model_path": str(model_path),
        "provider": provider,
        "batch_size": batch_size,
        "n_runs": n_runs,
        "avg_latency_ms": round(float(latencies.mean()), 2),
        "std_latency_ms": round(float(latencies.std()), 2),
        "min_latency_ms": round(float(latencies.min()), 2),
        "max_latency_ms": round(float(latencies.max()), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "throughput_patches_per_sec": round(float(batch_size * 1000 / latencies.mean()), 1),
    }

    logger.info(
        "Benchmark: avg=%.1fms, p95=%.1fms, throughput=%.0f patches/sec (%s)",
        results["avg_latency_ms"],
        results["p95_latency_ms"],
        results["throughput_patches_per_sec"],
        provider,
    )

    return results
