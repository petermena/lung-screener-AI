"""Model evaluation pipeline for lung nodule detection.

Runs comprehensive evaluation on the validation set and produces
a detailed report with clinical-grade metrics:
- AUC-ROC with confidence interval (bootstrap)
- Sensitivity / specificity at multiple operating points
- Precision / recall / F1
- Expected Calibration Error (ECE)
- Per-threshold operating point table
- Free-Response ROC (FROC) sensitivity at key false-positive rates
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from .calibration import _expected_calibration_error
from .dataset import CombinedLungDataset
from .model import build_model

logger = logging.getLogger(__name__)


def evaluate(
    config: dict,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Run full evaluation on the validation set.

    Args:
        config: Full configuration dict.
        checkpoint_path: Path to model checkpoint (.pth).
        output_path: Optional path to write JSON results.

    Returns:
        Dict with all evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # Build model and load weights
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    logger.info("Loaded checkpoint from epoch %s", epoch)

    # Create validation dataset
    val_dataset = CombinedLungDataset.from_config(config, split="val", augment=False)
    val_dataset.warm_disk_cache()
    logger.info("Validation samples: %d", len(val_dataset))

    train_config = config.get("training", {})
    num_workers = train_config.get("num_workers", 2)
    batch_size = train_config.get("batch_size", 24)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Collect predictions
    all_probs = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in val_loader:
            patches = batch["patch"].to(device)
            labels = batch["label"]

            with autocast("cuda", enabled=torch.cuda.is_available()):
                output = model(patches)

            logits = output["logits"].cpu().numpy()
            probs = torch.softmax(output["logits"], dim=1)[:, 1].cpu().numpy()

            all_logits.append(logits)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)

    num_pos = int(np.sum(all_labels == 1))
    num_neg = int(np.sum(all_labels == 0))
    logger.info("Positives: %d, Negatives: %d", num_pos, num_neg)

    # Core metrics at default threshold (0.5)
    threshold = config.get("inference", {}).get("threshold", 0.5)
    preds = (all_probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUC-ROC
    auc = _compute_auc(all_labels, all_probs)

    # Bootstrap 95% CI for AUC
    auc_ci_low, auc_ci_high = _bootstrap_auc_ci(all_labels, all_probs)

    # Expected Calibration Error
    ece = float(_expected_calibration_error(all_probs, all_labels))

    # Operating point table at multiple thresholds
    operating_points = _operating_points(all_labels, all_probs)

    # FROC-style: sensitivity at fixed false-positive rates
    froc_points = _froc_sensitivity(all_labels, all_probs)

    # Youden's J statistic (optimal threshold)
    best_threshold, best_j = _youden_optimal(all_labels, all_probs)

    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "num_samples": len(all_labels),
        "num_positives": num_pos,
        "num_negatives": num_neg,
        "threshold": threshold,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "metrics": {
            "auc_roc": round(auc, 4),
            "auc_95ci_low": round(auc_ci_low, 4),
            "auc_95ci_high": round(auc_ci_high, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "npv": round(npv, 4),
            "ece": round(ece, 4),
        },
        "optimal_threshold": {
            "threshold": round(best_threshold, 4),
            "youden_j": round(best_j, 4),
        },
        "operating_points": operating_points,
        "froc_sensitivity": froc_points,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results written to %s", output_path)

    return results


def _compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUC-ROC via trapezoidal rule."""
    if len(np.unique(labels)) < 2:
        return 0.0

    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]

    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    if num_pos == 0 or num_neg == 0:
        return 0.0

    tp_rate = np.cumsum(sorted_labels) / num_pos
    fp_rate = np.cumsum(1 - sorted_labels) / num_neg

    tp_rate = np.concatenate([[0], tp_rate])
    fp_rate = np.concatenate([[0], fp_rate])

    return float(_trapezoid(tp_rate, fp_rate))


def _bootstrap_auc_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(42)
    aucs = []
    n = len(labels)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_labels = labels[idx]
        boot_probs = probs[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        aucs.append(_compute_auc(boot_labels, boot_probs))

    if not aucs:
        return 0.0, 0.0

    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


def _operating_points(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    """Compute metrics at multiple threshold operating points."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    points = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        points.append({
            "threshold": t,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return points


def _froc_sensitivity(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute sensitivity at clinically-relevant false positive rates.

    Reports sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP per scan
    (approximated as FP rate since we're evaluating candidates, not scans).
    """
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    if num_pos == 0 or num_neg == 0:
        return {}

    sorted_indices = np.argsort(-probs)
    sorted_labels = labels[sorted_indices]

    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(1 - sorted_labels)

    fp_rates = fp_cumsum / num_neg
    sensitivities = tp_cumsum / num_pos

    target_fp_rates = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    result = {}
    for target in target_fp_rates:
        idx = np.searchsorted(fp_rates, target)
        if idx < len(sensitivities):
            result[f"sens_at_fpr_{target}"] = round(float(sensitivities[idx]), 4)
        else:
            result[f"sens_at_fpr_{target}"] = round(float(sensitivities[-1]), 4)

    return result


def _youden_optimal(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    best_j = -1.0
    best_t = 0.5
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)

    if num_pos == 0 or num_neg == 0:
        return 0.5, 0.0

    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        sens = tp / num_pos
        spec = tn / num_neg
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = t

    return float(best_t), float(best_j)


def format_report(results: dict) -> str:
    """Format evaluation results as a human-readable report."""
    m = results["metrics"]
    cm = results["confusion_matrix"]
    opt = results["optimal_threshold"]

    lines = []
    lines.append("=" * 60)
    lines.append("  LUNG SCREENER AI - EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Checkpoint:  {results['checkpoint']}")
    lines.append(f"  Epoch:       {results['epoch']}")
    lines.append(f"  Samples:     {results['num_samples']} "
                 f"({results['num_positives']} pos / {results['num_negatives']} neg)")
    lines.append(f"  Threshold:   {results['threshold']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  PRIMARY METRICS")
    lines.append("-" * 60)
    lines.append(f"  AUC-ROC:       {m['auc_roc']:.4f}  "
                 f"(95% CI: {m['auc_95ci_low']:.4f} - {m['auc_95ci_high']:.4f})")
    lines.append(f"  Sensitivity:   {m['sensitivity']:.4f}  (recall)")
    lines.append(f"  Specificity:   {m['specificity']:.4f}")
    lines.append(f"  Precision:     {m['precision']:.4f}  (PPV)")
    lines.append(f"  NPV:           {m['npv']:.4f}")
    lines.append(f"  F1 Score:      {m['f1_score']:.4f}")
    lines.append(f"  Accuracy:      {m['accuracy']:.4f}")
    lines.append(f"  ECE:           {m['ece']:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  CONFUSION MATRIX")
    lines.append("-" * 60)
    lines.append(f"                  Predicted +   Predicted -")
    lines.append(f"  Actual +          {cm['tp']:>5}         {cm['fn']:>5}")
    lines.append(f"  Actual -          {cm['fp']:>5}         {cm['tn']:>5}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  OPTIMAL OPERATING POINT (Youden's J)")
    lines.append("-" * 60)
    lines.append(f"  Threshold:     {opt['threshold']:.4f}")
    lines.append(f"  Youden's J:    {opt['youden_j']:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  OPERATING POINTS")
    lines.append("-" * 60)
    lines.append(f"  {'Thresh':>7}  {'Sens':>7}  {'Spec':>7}  {'Prec':>7}  "
                 f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}")
    for op in results["operating_points"]:
        lines.append(
            f"  {op['threshold']:>7.2f}  "
            f"{op['sensitivity']:>7.4f}  "
            f"{op['specificity']:>7.4f}  "
            f"{op['precision']:>7.4f}  "
            f"{op['tp']:>5}  {op['fp']:>5}  {op['fn']:>5}  {op['tn']:>5}"
        )
    lines.append("")

    if results.get("froc_sensitivity"):
        lines.append("-" * 60)
        lines.append("  FROC SENSITIVITY AT FALSE-POSITIVE RATES")
        lines.append("-" * 60)
        for key, val in results["froc_sensitivity"].items():
            fpr = key.replace("sens_at_fpr_", "FPR=")
            lines.append(f"  {fpr:<15}  Sensitivity: {val:.4f}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
