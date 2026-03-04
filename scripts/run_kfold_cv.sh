#!/usr/bin/env bash
# =============================================================================
# Step 2: K-Fold Cross-Validation Training (5 full training runs)
#
# Trains 5 independent models — each validated on a different 20% fold.
# Produces ensemble checkpoints and robust performance estimates.
#
# Outputs:
#   - checkpoints/fold_0/best.pth  ...  checkpoints/fold_4/best.pth
#   - checkpoints/kfold_summary.json   (mean/std AUC across all folds)
#
# Pass --resume to skip completed folds and resume interrupted ones.
#
# Estimated time: ~10-20 hours on g4dn.xlarge (T4) for 5 × 100 epochs
# Tip: run inside a screen/tmux session so you can detach safely.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Configurable options ---------------------------------------------------
FOLDS="${FOLDS:-5}"
RESUME_FLAG=""

# Pass RESUME=1 env var (or --resume arg) to resume interrupted runs
if [[ "${1:-}" == "--resume" ]] || [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
    echo "[INFO] Resume mode: completed folds will be skipped."
fi
# -----------------------------------------------------------------------------

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/kfold_cv_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================================="
echo "  K-FOLD CROSS-VALIDATION TRAINING — $(date)"
echo "=================================================================="
echo "  Project:    ${PROJECT_ROOT}"
echo "  Folds:      ${FOLDS}"
echo "  Resume:     ${RESUME_FLAG:-no}"
echo "  Log file:   ${LOGFILE}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "=================================================================="
echo ""
echo "  NOTE: This will run ${FOLDS} full training passes."
echo "  Each fold takes ~2-4 h on a T4 GPU — total ~10-20 h."
echo "  Run inside screen/tmux to detach:  screen -S kfold"
echo "=================================================================="
echo ""

lung-screener train-kfold \
    --folds "${FOLDS}" \
    --checkpoint-dir checkpoints \
    ${RESUME_FLAG} \
    2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "=================================================================="
    echo "  K-FOLD TRAINING COMPLETE — $(date)"
    echo "  Summary:  checkpoints/kfold_summary.json"
    echo "  Full log: ${LOGFILE}"
    echo "=================================================================="

    echo ""
    echo "--- K-Fold summary ---"
    python3 - <<'EOF'
import json, pathlib
p = pathlib.Path("checkpoints/kfold_summary.json")
if not p.exists():
    print("  (summary file not found)")
else:
    s = json.loads(p.read_text())
    print(f"  Folds: {s['n_folds']}")
    print(f"  Mean AUC: {s['mean_auc']:.4f} ± {s['std_auc']:.4f}")
    print(f"  Range:    {s['min_auc']:.4f} – {s['max_auc']:.4f}")
    for f in s["folds"]:
        print(f"    fold {f['fold']}: AUC={f['best_val_auc']:.4f}  ckpt={f['checkpoint']}")
EOF

    echo ""
    echo "--- Evaluate ensemble on validation set ---"
    echo "  Run: lung-screener evaluate-kfold -k ${FOLDS} --checkpoint-dir checkpoints"
else
    echo "=================================================================="
    echo "  K-FOLD TRAINING FAILED (exit code ${EXIT_CODE}) — $(date)"
    echo "  To resume:  RESUME=1 bash scripts/run_kfold_cv.sh"
    echo "  Check log:  ${LOGFILE}"
    echo "=================================================================="
    exit ${EXIT_CODE}
fi
