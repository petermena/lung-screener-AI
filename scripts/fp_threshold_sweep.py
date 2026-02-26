#!/usr/bin/env python3
"""Sweep FP-reduction thresholds without retraining.

Loads both the first-stage and FP-reduction checkpoints, then evaluates
the two-stage pipeline at multiple FP thresholds to find the operating
point that best preserves sensitivity while still reducing false positives.

Usage:
    python scripts/fp_threshold_sweep.py
    python scripts/fp_threshold_sweep.py --thresholds 0.2 0.3 0.4 0.5
    python scripts/fp_threshold_sweep.py --first-stage-ckpt checkpoints/best.pth \
        --fp-ckpt checkpoints/fp_reduction/best.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lung_screener.dataset import CombinedLungDataset
from lung_screener.fp_reduction import FPReductionNet
from lung_screener.model import build_model
from train_fp_reduction import evaluate_two_stage, load_config, format_two_stage_report

logger = logging.getLogger(__name__)


def sweep_thresholds(
    first_stage_model,
    fp_model,
    val_loader,
    device,
    stage1_threshold: float,
    fp_thresholds: list[float],
) -> list[dict]:
    """Run evaluate_two_stage at each FP threshold and collect results."""
    results = []
    for thr in fp_thresholds:
        logger.info("Evaluating FP threshold = %.2f ...", thr)
        r = evaluate_two_stage(
            first_stage_model=first_stage_model,
            fp_model=fp_model,
            dataloader=val_loader,
            device=device,
            stage1_threshold=stage1_threshold,
            fp_threshold=thr,
        )
        results.append(r)
    return results


def print_sweep_table(results: list[dict]) -> None:
    """Print a compact comparison table across all thresholds."""
    header = (
        f"\n{'FP thr':>7}  {'Sens':>6}  {'Spec':>6}  {'Prec':>6}  "
        f"{'F1':>6}  {'AUC':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  "
        f"{'FP-red%':>7}  {'dSens':>7}"
    )
    divider = "-" * len(header.lstrip("\n"))
    print("\n" + "=" * len(header.lstrip("\n")))
    print("  FP THRESHOLD SWEEP — TWO-STAGE PIPELINE")
    print("=" * len(header.lstrip("\n")))
    print(header)
    print(divider)

    s1 = results[0]["stage1_only"]
    print(
        f"  {'STAGE1':>5}  {s1['sensitivity']:>6.4f}  {s1['specificity']:>6.4f}  "
        f"{s1['precision']:>6.4f}  {s1['f1']:>6.4f}  {s1['auc_roc']:>6.4f}  "
        f"{s1['tp']:>5}  {s1['fp']:>5}  {s1['fn']:>5}  "
        f"{'—':>7}  {'baseline':>7}"
    )
    print(divider)

    best_f1_thr = None
    best_f1 = -1.0
    best_clinical_thr = None  # highest FP threshold with sens >= 0.90

    for r in results:
        ts = r["two_stage"]
        imp = r["improvement"]
        thr = r["fp_threshold"]
        print(
            f"  {thr:>7.2f}  {ts['sensitivity']:>6.4f}  {ts['specificity']:>6.4f}  "
            f"{ts['precision']:>6.4f}  {ts['f1']:>6.4f}  {ts['auc_roc']:>6.4f}  "
            f"{ts['tp']:>5}  {ts['fp']:>5}  {ts['fn']:>5}  "
            f"{imp['fp_reduction_pct']:>6.1f}%  {imp['sensitivity_delta']:>+7.4f}"
        )
        if ts["f1"] > best_f1:
            best_f1 = ts["f1"]
            best_f1_thr = thr
        if ts["sensitivity"] >= 0.90:
            best_clinical_thr = thr  # keep updating; last one with >= 0.90 sens is highest threshold

    print(divider)
    print(f"\n  Best F1 threshold:              {best_f1_thr:.2f}  (F1={best_f1:.4f})")
    if best_clinical_thr is not None:
        print(f"  Highest thr with sens >= 0.90:  {best_clinical_thr:.2f}")
    else:
        print("  No threshold achieved sens >= 0.90 — consider using stage-1 alone")
    print()


def main():
    parser = argparse.ArgumentParser(description="Sweep FP reduction thresholds")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="FP thresholds to evaluate (default: 0.10 to 0.50 in 0.05 steps)",
    )
    parser.add_argument(
        "--stage1-threshold", type=float, default=None,
        help="First-stage threshold (default: from config)",
    )
    parser.add_argument(
        "--first-stage-ckpt", type=str, default=None,
        help="Path to first-stage best.pth",
    )
    parser.add_argument(
        "--fp-ckpt", type=str, default=None,
        help="Path to FP reduction best.pth",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Base checkpoint directory",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Override config YAML",
    )
    parser.add_argument(
        "--batch-size", type=int, default=96,
        help="Batch size for evaluation (default: 96)",
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Optional path to save sweep results as JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1_threshold = args.stage1_threshold or config.get("inference", {}).get("threshold", 0.15)
    first_stage_path = args.first_stage_ckpt or str(checkpoint_dir / "best.pth")
    fp_ckpt_path = args.fp_ckpt or str(checkpoint_dir / "fp_reduction" / "best.pth")

    logger.info("First-stage checkpoint: %s", first_stage_path)
    logger.info("FP reduction checkpoint: %s", fp_ckpt_path)
    logger.info("Stage-1 threshold: %.3f", stage1_threshold)
    logger.info("FP thresholds to sweep: %s", args.thresholds)
    logger.info("Device: %s", device)

    # Load first-stage model
    logger.info("Loading first-stage model...")
    first_stage_model = build_model(config).to(device)
    ckpt = torch.load(first_stage_path, map_location=device, weights_only=False)
    first_stage_model.load_state_dict(
        ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    )
    first_stage_model.eval()
    logger.info("  Loaded (epoch %s)", ckpt.get("epoch", "?"))

    # Load FP reduction model
    logger.info("Loading FP reduction model...")
    fp_model = FPReductionNet(in_channels=1, base_filters=32).to(device)
    fp_ckpt = torch.load(fp_ckpt_path, map_location=device, weights_only=False)
    fp_model.load_state_dict(fp_ckpt["model_state_dict"])
    fp_model.eval()
    logger.info("  Loaded (best val AUC: %.4f)", fp_ckpt.get("best_val_auc", 0.0))

    # Load validation dataset (once, shared across all threshold evaluations)
    logger.info("Loading validation dataset...")
    val_dataset = CombinedLungDataset.from_config(config, split="val", augment=False)
    val_dataset.warm_disk_cache()
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("training", {}).get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    logger.info("  Val candidates: %d", len(val_dataset))

    # Sweep thresholds
    results = sweep_thresholds(
        first_stage_model=first_stage_model,
        fp_model=fp_model,
        val_loader=val_loader,
        device=device,
        stage1_threshold=stage1_threshold,
        fp_thresholds=sorted(args.thresholds),
    )

    # Print compact table
    print_sweep_table(results)

    # Optionally print full report for the best clinical threshold
    s1_sens = results[0]["stage1_only"]["sensitivity"]
    best_clinical = None
    for r in results:
        if r["two_stage"]["sensitivity"] >= 0.90:
            best_clinical = r
    if best_clinical is None:
        # Fall back to highest sensitivity threshold
        best_clinical = max(results, key=lambda r: r["two_stage"]["sensitivity"])

    print(f"\nFull report for recommended threshold ({best_clinical['fp_threshold']:.2f}):")
    print(format_two_stage_report(best_clinical))

    # Save JSON if requested
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Sweep results saved to %s", args.save_json)


if __name__ == "__main__":
    main()
