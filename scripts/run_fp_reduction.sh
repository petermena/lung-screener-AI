#!/usr/bin/env bash
# =============================================================================
# Step 1: FP Reduction Training (second-stage hard-negative mining)
#
# Requires:
#   - checkpoints/best.pth   (first-stage checkpoint from main training)
#   - data/luna16 and/or data/luna25  (same data used in first-stage training)
#
# Outputs:
#   - checkpoints/fp_reduction/best.pth
#   - checkpoints/fp_reduction/fp_training_history.json
#   - checkpoints/fp_reduction/eval_results.json
#   - checkpoints/fp_reduction/fp_reduction_config.yaml
#
# Estimated time: 2-4 hours on g4dn.xlarge (T4)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/fp_reduction_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================================="
echo "  FP REDUCTION TRAINING — $(date)"
echo "=================================================================="
echo "  Project:    ${PROJECT_ROOT}"
echo "  Log file:   ${LOGFILE}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "=================================================================="

# Verify first-stage checkpoint exists
FIRST_STAGE_CKPT="${PROJECT_ROOT}/checkpoints/best.pth"
if [[ ! -f "${FIRST_STAGE_CKPT}" ]]; then
    echo "ERROR: First-stage checkpoint not found at ${FIRST_STAGE_CKPT}"
    echo "       Run main training first before FP reduction."
    exit 1
fi

echo ""
echo "[$(date +%H:%M:%S)] Starting FP reduction pipeline..."
echo ""

python scripts/train_fp_reduction.py \
    --config config/fp_reduction.yaml \
    --first-stage-ckpt "${FIRST_STAGE_CKPT}" \
    --checkpoint-dir checkpoints \
    --hard-negative-ratio 3.0 \
    --score-batch-size 96 \
    --verbose \
    2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "=================================================================="
    echo "  FP REDUCTION COMPLETE — $(date)"
    echo "  Best checkpoint: checkpoints/fp_reduction/best.pth"
    echo "  Full log:        ${LOGFILE}"
    echo "=================================================================="

    # Print the two-stage eval summary from the end of the log
    echo ""
    echo "--- Two-stage evaluation summary ---"
    grep -A 40 "TWO-STAGE PIPELINE EVALUATION" "${LOGFILE}" | tail -42 || true
else
    echo "=================================================================="
    echo "  FP REDUCTION FAILED (exit code ${EXIT_CODE}) — $(date)"
    echo "  Check log: ${LOGFILE}"
    echo "=================================================================="
    exit ${EXIT_CODE}
fi
