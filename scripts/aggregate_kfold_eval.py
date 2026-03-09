#!/usr/bin/env python3
"""Aggregate k-fold evaluation results into a single summary.

Runs evaluate.py on each fold checkpoint (if not already done), then
computes mean ± std across folds for every metric.

Usage:
    python3 scripts/aggregate_kfold_eval.py
    python3 scripts/aggregate_kfold_eval.py --n-folds 5 --checkpoint-dir checkpoints
    python3 scripts/aggregate_kfold_eval.py --skip-eval   # only aggregate existing fold eval_results.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = [
    "auc_roc",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "npv",
    "ece",
]

FROC_KEYS = [
    "sens_at_fpr_0.0125",
    "sens_at_fpr_0.025",
    "sens_at_fpr_0.05",
    "sens_at_fpr_0.1",
    "sens_at_fpr_0.2",
    "sens_at_fpr_0.4",
    "sens_at_fpr_0.8",
]


def run_eval(fold_dir: Path, config: str | None) -> Path:
    """Run evaluate.py for a single fold and return path to its results JSON."""
    out = fold_dir / "eval_results.json"
    if out.exists():
        print(f"  [skip] {fold_dir.name}: eval_results.json already exists")
        return out

    cmd = [
        sys.executable,
        "evaluate.py",
        "--checkpoint", str(fold_dir / "best.pth"),
        "--output", str(out),
    ]
    if config:
        cmd += ["--config", config]

    print(f"  [eval] {fold_dir.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return out


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def aggregate(fold_results: list[dict]) -> dict:
    """Compute mean ± std across folds for all standard metrics."""
    agg: dict = {"n_folds": len(fold_results), "folds": [], "metrics": {}, "froc_sensitivity": {}}

    # Per-fold summary
    for i, r in enumerate(fold_results):
        agg["folds"].append({
            "fold": i,
            "checkpoint": r.get("checkpoint", f"fold_{i}/best.pth"),
            "num_samples": r.get("num_samples"),
            "threshold": r.get("threshold"),
            **{k: r["metrics"][k] for k in METRICS if k in r.get("metrics", {})},
        })

    # Aggregate scalar metrics
    for key in METRICS:
        values = [r["metrics"][key] for r in fold_results if key in r.get("metrics", {})]
        if not values:
            continue
        arr = np.array(values)
        agg["metrics"][key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": [float(v) for v in arr],
        }

    # AUC 95% CI: pool the per-fold CI widths as a rough estimate
    auc_lows = [r["metrics"].get("auc_95ci_low") for r in fold_results]
    auc_highs = [r["metrics"].get("auc_95ci_high") for r in fold_results]
    if all(v is not None for v in auc_lows + auc_highs):
        mean_auc = agg["metrics"]["auc_roc"]["mean"]
        half_width = np.mean([h - l for h, l in zip(auc_highs, auc_lows)]) / 2
        agg["metrics"]["auc_95ci_low"] = float(mean_auc - half_width)
        agg["metrics"]["auc_95ci_high"] = float(mean_auc + half_width)

    # Aggregate FROC sensitivity
    for key in FROC_KEYS:
        values = [r.get("froc_sensitivity", {}).get(key) for r in fold_results]
        values = [v for v in values if v is not None]
        if values:
            arr = np.array(values)
            agg["froc_sensitivity"][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": [float(v) for v in arr],
            }

    return agg


def print_report(agg: dict) -> None:
    m = agg["metrics"]
    n = agg["n_folds"]
    print(f"\n{'='*55}")
    print(f"  K-Fold Evaluation Summary  ({n} folds)")
    print(f"{'='*55}")
    for key in METRICS:
        if key not in m:
            continue
        d = m[key]
        label = key.upper().replace("_", " ")
        print(f"  {label:<20} {d['mean']:.4f} ± {d['std']:.4f}   [{d['min']:.4f} – {d['max']:.4f}]")

    if "auc_95ci_low" in m:
        print(f"  {'AUC 95% CI':<20} [{m['auc_95ci_low']:.4f} – {m['auc_95ci_high']:.4f}]")

    if agg["froc_sensitivity"]:
        print(f"\n  FROC Sensitivity (mean ± std):")
        for key, d in agg["froc_sensitivity"].items():
            fpr = key.replace("sens_at_fpr_", "FPR=")
            print(f"    {fpr:<12} {d['mean']:.4f} ± {d['std']:.4f}")

    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate k-fold evaluation results.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--config", default=None, help="Config YAML passed to evaluate.py")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluate.py; only aggregate existing eval_results.json files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write aggregated JSON (default: <checkpoint-dir>/kfold_eval_summary.json)",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_path = Path(args.output) if args.output else checkpoint_dir / "kfold_eval_summary.json"

    fold_dirs = [checkpoint_dir / f"fold_{i}" for i in range(args.n_folds)]
    missing = [d for d in fold_dirs if not d.exists()]
    if missing:
        print(f"ERROR: fold directories not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {args.n_folds} folds in {checkpoint_dir}/")
    result_paths = []
    for fold_dir in fold_dirs:
        if args.skip_eval:
            p = fold_dir / "eval_results.json"
            if not p.exists():
                print(f"ERROR: {p} not found. Run without --skip-eval first.", file=sys.stderr)
                sys.exit(1)
            result_paths.append(p)
        else:
            result_paths.append(run_eval(fold_dir, args.config))

    fold_results = [load_results(p) for p in result_paths]
    agg = aggregate(fold_results)

    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    print_report(agg)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
