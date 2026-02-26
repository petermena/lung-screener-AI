"""End-to-end model pipeline: evaluate, dashboard, calibrate, optimize, export, k-fold.

Orchestrates the full post-training workflow:
1. Evaluate model on validation set (comprehensive metrics)
2. Generate interactive metrics dashboard
3. Calibrate confidence scores (temperature scaling)
4. Export to ONNX for lightweight deployment
5. Optimize ONNX model (graph optimization + INT8 quantization)
6. Benchmark inference latency
7. (Optional) K-fold cross-validation training

All results are saved to the checkpoint directory and the dashboard
is regenerated with each step's output.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(
    config: dict,
    checkpoint_path: str | Path,
    checkpoint_dir: str | Path = "./checkpoints",
    steps: list[str] | None = None,
    kfold_folds: int = 5,
) -> dict:
    """Run the full post-training pipeline.

    Args:
        config: Full configuration dict.
        checkpoint_path: Path to trained model checkpoint (.pth).
        checkpoint_dir: Directory for all outputs.
        steps: List of steps to run. Defaults to all.
            Available: "evaluate", "dashboard", "calibrate",
                       "optimize", "export", "kfold"
        kfold_folds: Number of folds for k-fold (if included).

    Returns:
        Dict with results from each completed step.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_steps = ["evaluate", "dashboard", "calibrate", "optimize", "export", "kfold"]
    if steps is None:
        steps = ["evaluate", "dashboard", "calibrate", "export", "optimize"]

    results = {}

    # ================================================================
    # Step 1: Evaluate
    # ================================================================
    if "evaluate" in steps:
        logger.info("=" * 60)
        logger.info("  STEP 1: Model Evaluation")
        logger.info("=" * 60)

        from .evaluate import evaluate_full, format_report

        eval_output = checkpoint_dir / "eval_results.json"
        eval_results = evaluate_full(config, checkpoint_path, output_path=eval_output)
        results["evaluate"] = eval_results

        # Print report
        from .evaluate import format_report
        logger.info("\n%s", format_report(eval_results))

    # ================================================================
    # Step 2: Calibrate
    # ================================================================
    if "calibrate" in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STEP 2: Confidence Calibration")
        logger.info("=" * 60)

        import numpy as np
        import torch
        from torch.amp import autocast
        from torch.utils.data import DataLoader

        from .calibration import TemperatureScaler, PlattScaler, _expected_calibration_error
        from .dataset import CombinedLungDataset
        from .model import build_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model = build_model(config, state_dict=state_dict).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Collect logits
        val_dataset = CombinedLungDataset.from_config(config, split="val", augment=False)
        val_dataset.warm_disk_cache()
        train_config = config.get("training", {})
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.get("batch_size", 24),
            shuffle=False,
            num_workers=train_config.get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
        )

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                patches = batch["patch"].to(device)
                labels = batch["label"]
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    model_out = model(patches)
                all_logits.append(model_out["logits"].cpu().numpy())
                all_labels.extend(labels.numpy())

        logits = np.concatenate(all_logits, axis=0)
        labels = np.array(all_labels)

        # Pre-calibration ECE
        probs_before = torch.softmax(torch.from_numpy(logits).float(), dim=1)[:, 1].numpy()
        ece_before = _expected_calibration_error(probs_before, labels)

        # Temperature scaling
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(logits, labels)
        probs_temp = temp_scaler.calibrate(logits)
        ece_temp = _expected_calibration_error(probs_temp, labels)

        # Platt scaling
        platt_scaler = PlattScaler()
        platt_scaler.fit(probs_before, labels)
        probs_platt = platt_scaler.calibrate(probs_before)
        ece_platt = _expected_calibration_error(probs_platt, labels)

        # Pick the best method
        if ece_temp <= ece_platt:
            best_method = "temperature"
            best_ece = ece_temp
            temp_scaler.save(checkpoint_dir / "calibration.json")
        else:
            best_method = "platt"
            best_ece = ece_platt
            platt_scaler.save(checkpoint_dir / "calibration.json")

        cal_results = {
            "ece_before": round(float(ece_before), 4),
            "temperature_scaling": {
                "temperature": round(temp_scaler.temperature, 4),
                "ece_after": round(float(ece_temp), 4),
            },
            "platt_scaling": {
                "a": round(platt_scaler.a, 4),
                "b": round(platt_scaler.b, 4),
                "ece_after": round(float(ece_platt), 4),
            },
            "best_method": best_method,
            "best_ece": round(float(best_ece), 4),
            "improvement": round(float(ece_before - best_ece), 4),
        }
        results["calibrate"] = cal_results

        with open(checkpoint_dir / "calibration_results.json", "w") as f:
            json.dump(cal_results, f, indent=2)

        logger.info("Calibration complete:")
        logger.info("  ECE before:           %.4f", ece_before)
        logger.info("  ECE (temperature):    %.4f (T=%.4f)",
                     ece_temp, temp_scaler.temperature)
        logger.info("  ECE (Platt):          %.4f (A=%.4f, B=%.4f)",
                     ece_platt, platt_scaler.a, platt_scaler.b)
        logger.info("  Best method:          %s (ECE=%.4f)", best_method, best_ece)

    # ================================================================
    # Step 3: Export to ONNX
    # ================================================================
    if "export" in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STEP 3: Export to ONNX")
        logger.info("=" * 60)

        from .export import export_to_onnx

        onnx_path = checkpoint_dir / "model.onnx"
        export_to_onnx(checkpoint_path, onnx_path, config)
        results["export"] = {"onnx_path": str(onnx_path)}
        logger.info("Exported to %s", onnx_path)

    # ================================================================
    # Step 4: Optimize ONNX
    # ================================================================
    if "optimize" in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STEP 4: ONNX Optimization + Quantization")
        logger.info("=" * 60)

        from .optimize import benchmark_onnx, optimize_onnx

        onnx_path = checkpoint_dir / "model.onnx"
        if not onnx_path.exists():
            # Export first if not already done
            from .export import export_to_onnx
            export_to_onnx(checkpoint_path, onnx_path, config)

        # Graph optimization only
        opt_path = checkpoint_dir / "model_optimized.onnx"
        opt_results = optimize_onnx(onnx_path, opt_path, quantize=False)

        # Graph optimization + INT8 quantization
        q8_path = checkpoint_dir / "model_optimized_q8.onnx"
        q8_results = optimize_onnx(onnx_path, q8_path, quantize=True)

        # Benchmark all variants
        patch_size = tuple(config.get("model", {}).get("patch_size", [48, 48, 48]))
        logger.info("Benchmarking inference latency...")

        benchmarks = {}
        for name, path in [("original", onnx_path), ("optimized", opt_path), ("quantized_int8", q8_path)]:
            if path.exists():
                bm = benchmark_onnx(path, patch_size=patch_size)
                benchmarks[name] = bm

        optimization_results = {
            "original_size_mb": opt_results["original_size_mb"],
            "optimized_size_mb": opt_results["optimized_size_mb"],
            "quantized_size_mb": q8_results["optimized_size_mb"],
            "graph_optimized": True,
            "quantized": True,
            "reduction_pct": q8_results["reduction_pct"],
            "output_path": str(q8_path),
            "benchmark": benchmarks,
        }
        results["optimize"] = optimization_results

        with open(checkpoint_dir / "optimization_results.json", "w") as f:
            json.dump(optimization_results, f, indent=2)

        logger.info("Optimization results:")
        logger.info("  Original:    %.1f MB", opt_results["original_size_mb"])
        logger.info("  Optimized:   %.1f MB", opt_results["optimized_size_mb"])
        logger.info("  Quantized:   %.1f MB", q8_results["optimized_size_mb"])
        for name, bm in benchmarks.items():
            logger.info("  %s: avg=%.1fms, p95=%.1fms, %.0f patches/sec",
                         name, bm["avg_latency_ms"], bm["p95_latency_ms"],
                         bm["throughput_patches_per_sec"])

    # ================================================================
    # Step 5: Generate Dashboard
    # ================================================================
    if "dashboard" in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STEP 5: Generate Comprehensive Dashboard")
        logger.info("=" * 60)

        from .metrics_dashboard import save_dashboard

        metrics_path = checkpoint_dir / "metrics.json"
        dashboard_path = save_dashboard(
            metrics_path,
            output_path=checkpoint_dir / "dashboard.html",
            eval_results_path=checkpoint_dir / "eval_results.json",
            calibration_path=checkpoint_dir / "calibration.json",
            optimization_path=checkpoint_dir / "optimization_results.json",
        )
        results["dashboard"] = {"path": str(dashboard_path)}
        logger.info("Dashboard saved to %s", dashboard_path)

    # ================================================================
    # Step 6: K-Fold Cross-Validation
    # ================================================================
    if "kfold" in steps:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  STEP 6: K-Fold Cross-Validation (%d folds)", kfold_folds)
        logger.info("=" * 60)

        from .train import train_kfold

        kfold_summary = train_kfold(
            config, n_folds=kfold_folds, checkpoint_dir=str(checkpoint_dir),
        )
        results["kfold"] = kfold_summary
        logger.info(
            "K-Fold: AUC = %.4f +/- %.4f",
            kfold_summary["mean_auc"], kfold_summary["std_auc"],
        )

    # ================================================================
    # Final Summary
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Steps completed: %s", ", ".join(steps))
    logger.info("Output directory: %s", checkpoint_dir)

    # Save pipeline summary
    summary_path = checkpoint_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        # Serialize only JSON-safe data
        safe_results = {}
        for k, v in results.items():
            try:
                json.dumps(v)
                safe_results[k] = v
            except (TypeError, ValueError):
                safe_results[k] = str(v)
        json.dump(safe_results, f, indent=2)

    return results
