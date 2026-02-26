#!/usr/bin/env python3
"""Top-level convenience script for model evaluation.

Usage:
    python3 evaluate.py --checkpoint checkpoints/best.pth
    python3 evaluate.py --checkpoint checkpoints/best.pth --output checkpoints/eval_results.json
    python3 evaluate.py --checkpoint checkpoints/best.pth --output checkpoints/eval_results.json --full
    python3 evaluate.py --checkpoint checkpoints/best.pth --config config/custom.yaml --output results.json
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from lung_screener.cli import load_config
from lung_screener.evaluate import evaluate, evaluate_full, format_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a lung-nodule detection checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-m",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to write JSON results (e.g. checkpoints/eval_results.json)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Optional config YAML to override defaults",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include ROC/PR curve data and histograms (needed for dashboard)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.full:
        results = evaluate_full(config, args.checkpoint, output_path=args.output)
    else:
        results = evaluate(config, args.checkpoint, output_path=args.output)

    print(format_report(results))


if __name__ == "__main__":
    main()
