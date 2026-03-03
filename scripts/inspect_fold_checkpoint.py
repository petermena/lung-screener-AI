#!/usr/bin/env python3
"""Inspect a fold checkpoint and print its stored metadata.

Usage:
    python scripts/inspect_fold_checkpoint.py checkpoints/fold_0/best.pth
    python scripts/inspect_fold_checkpoint.py checkpoints/fold_0/best.pth --json
    python scripts/inspect_fold_checkpoint.py checkpoints/  # scan all folds
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def inspect_checkpoint(path: Path) -> dict:
    """Load a checkpoint and return its non-weight metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    meta = {}
    skip_keys = {"model_state_dict", "optimizer_state_dict", "scaler_state_dict",
                 "scheduler_state_dict", "swa_state_dict"}

    for k, v in ckpt.items():
        if k in skip_keys:
            continue
        # Flatten nested dicts up to one level
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if not isinstance(sub_v, (dict, list)) or len(str(sub_v)) < 200:
                    meta[f"{k}.{sub_k}"] = sub_v
        else:
            meta[k] = v

    return meta


def scan_fold_dir(checkpoint_dir: Path) -> list[dict]:
    """Find all fold checkpoints under a directory and inspect them."""
    results = []
    for fold_dir in sorted(checkpoint_dir.glob("fold_*")):
        for ckpt_name in ("best.pth", "latest.pth"):
            ckpt_path = fold_dir / ckpt_name
            if ckpt_path.exists():
                meta = inspect_checkpoint(ckpt_path)
                results.append({"path": str(ckpt_path), **meta})
                break  # prefer best over latest
    return results


def print_summary(path: str, meta: dict):
    print(f"\n{'='*60}")
    print(f"  Checkpoint: {path}")
    print(f"{'='*60}")

    key_fields = [
        ("epoch",          "Epoch"),
        ("total_epochs",   "Total epochs"),
        ("phase",          "Phase"),
        ("best_val_auc",   "Best val AUC"),
        ("metrics.auc",    "AUC (metrics)"),
        ("metrics.sensitivity", "Sensitivity"),
        ("metrics.specificity", "Specificity"),
        ("metrics.f1_score",    "F1 score"),
        ("fold",           "Fold index"),
        ("timestamp",      "Timestamp"),
    ]

    for key, label in key_fields:
        val = meta.get(key)
        if val is not None:
            if isinstance(val, float):
                print(f"  {label:<22}: {val:.4f}")
            else:
                print(f"  {label:<22}: {val}")

    # Any extra keys not in the standard list
    shown = {k for k, _ in key_fields}
    extras = {k: v for k, v in meta.items() if k not in shown and not isinstance(v, dict)}
    if extras:
        print("  --- other fields ---")
        for k, v in extras.items():
            print(f"  {k:<22}: {v}")

    completed = meta.get("phase") == "complete"
    epoch = meta.get("epoch", "?")
    total = meta.get("total_epochs", "?")
    status = "COMPLETE" if completed else f"INCOMPLETE (epoch {epoch}/{total})"
    print(f"\n  Status: {status}")


def main():
    parser = argparse.ArgumentParser(description="Inspect fold checkpoint metadata")
    parser.add_argument("path", help="Path to .pth file or checkpoint root directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir():
        results = scan_fold_dir(target)
        if not results:
            print(f"No fold checkpoints found under {target}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for r in results:
                path = r.pop("path")
                print_summary(path, r)
    elif target.is_file():
        meta = inspect_checkpoint(target)
        if args.json:
            print(json.dumps(meta, indent=2, default=str))
        else:
            print_summary(str(target), meta)
    else:
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
