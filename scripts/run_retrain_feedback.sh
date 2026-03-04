#!/usr/bin/env bash
# =============================================================================
# Step 4: Retrain from Radiologist Feedback
#
# Fine-tunes the current best model using confirmed/corrected findings that
# radiologists have submitted via the annotator UI or PACS workflow.
#
# Prerequisites (ALL must be met before running):
#   - checkpoints/best.pth           (trained first-stage model)
#   - data/feedback/                 (radiologist feedback records)
#     Must contain at least MIN_FEEDBACK confirmed records (default: 20)
#
# What this script does:
#   1. Shows current feedback statistics
#   2. Checks the minimum-record threshold
#   3. Runs IncrementalRetrainer — fine-tunes on feedback + base data
#   4. Only promotes the new model if it doesn't regress on the held-out set
#
# Outputs:
#   - checkpoints/retrain_<id>/best.pth   (new fine-tuned model)
#   - checkpoints/retrain_<id>/retrain_result.json
#   - checkpoints/best.pth is REPLACED only if the new model is better
#
# To fill data/feedback/ first, run the annotator UI:
#   lung-screener annotate
# Or review statistics with:
#   lung-screener feedback
#
# Estimated time: 30-60 minutes on a T4 for ~30 fine-tuning epochs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Configurable options ---------------------------------------------------
CKPT="${CKPT:-checkpoints/best.pth}"
FEEDBACK_DIR="${FEEDBACK_DIR:-data/feedback}"
MIN_FEEDBACK="${MIN_FEEDBACK:-20}"
EPOCHS="${EPOCHS:-}"          # leave blank to use config default (30)
LR="${LR:-}"                  # leave blank to use config default
# -----------------------------------------------------------------------------

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/retrain_feedback_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================================="
echo "  RETRAIN FROM FEEDBACK — $(date)"
echo "=================================================================="
echo "  Checkpoint:    ${CKPT}"
echo "  Feedback dir:  ${FEEDBACK_DIR}"
echo "  Min feedback:  ${MIN_FEEDBACK} records"
echo "  Log file:      ${LOGFILE}"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU only')"
echo "=================================================================="

# ------------------------------------------------------------------
# 1. Verify checkpoint
# ------------------------------------------------------------------
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: Checkpoint not found at ${CKPT}"
    echo "       Train a first-stage model before retraining from feedback."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Verify feedback directory and record count
# ------------------------------------------------------------------
if [[ ! -d "${FEEDBACK_DIR}" ]]; then
    echo "ERROR: Feedback directory not found: ${FEEDBACK_DIR}"
    echo "       Collect radiologist feedback first:"
    echo "         lung-screener annotate"
    echo "         lung-screener feedback"
    exit 1
fi

echo ""
echo "[$(date +%H:%M:%S)] Current feedback statistics:"
lung-screener feedback --feedback-dir "${FEEDBACK_DIR}" 2>&1 | tee -a "${LOGFILE}"

# Count confirmed feedback records (JSON files with status=confirmed)
CONFIRMED_COUNT=$(python3 -c "
import json, pathlib, sys
fb_dir = pathlib.Path('${FEEDBACK_DIR}')
count = 0
for f in fb_dir.glob('*.json'):
    try:
        data = json.loads(f.read_text())
        # Support both list and single-record formats
        records = data if isinstance(data, list) else [data]
        for r in records:
            if r.get('status') in ('confirmed', 'corrected', 'rejected'):
                count += 1
    except Exception:
        pass
print(count)
" 2>/dev/null || echo "0")

echo ""
echo "[$(date +%H:%M:%S)] Confirmed/corrected feedback records found: ${CONFIRMED_COUNT}"

if [[ "${CONFIRMED_COUNT}" -lt "${MIN_FEEDBACK}" ]]; then
    echo ""
    echo "WARNING: Only ${CONFIRMED_COUNT} feedback records found."
    echo "         Minimum required: ${MIN_FEEDBACK}"
    echo ""
    echo "  Options:"
    echo "    1. Collect more feedback:  lung-screener annotate"
    echo "    2. Lower the threshold:    MIN_FEEDBACK=10 bash scripts/run_retrain_feedback.sh"
    echo "    3. Force run anyway:       MIN_FEEDBACK=0 bash scripts/run_retrain_feedback.sh"
    echo ""
    echo "  Skipping retrain — not enough feedback yet."
    exit 0
fi

# ------------------------------------------------------------------
# 3. Build CLI args
# ------------------------------------------------------------------
EXTRA_ARGS=""
if [[ -n "${EPOCHS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --epochs ${EPOCHS}"
fi
if [[ -n "${LR}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --lr ${LR}"
fi

echo ""
echo "[$(date +%H:%M:%S)] Starting incremental retrain with ${CONFIRMED_COUNT} feedback records..."
echo ""

lung-screener retrain \
    --checkpoint "${CKPT}" \
    --feedback-dir "${FEEDBACK_DIR}" \
    --checkpoint-dir checkpoints \
    --min-feedback "${MIN_FEEDBACK}" \
    ${EXTRA_ARGS} \
    2>&1 | tee -a "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "=================================================================="
    echo "  RETRAIN COMPLETE — $(date)"
    echo "  Full log: ${LOGFILE}"
    echo "=================================================================="

    # Show retrain history
    echo ""
    echo "--- Retrain history ---"
    lung-screener retrain \
        --checkpoint "${CKPT}" \
        --feedback-dir "${FEEDBACK_DIR}" \
        --checkpoint-dir checkpoints \
        --history 2>&1 | tail -20 | tee -a "${LOGFILE}"

    echo ""
    echo "  If the new model was promoted, checkpoints/best.pth has been updated."
    echo "  Rebuild Docker image to deploy the updated model:"
    echo "    bash scripts/run_docker_deploy.sh"
else
    echo "=================================================================="
    echo "  RETRAIN FAILED (exit code ${EXIT_CODE}) — $(date)"
    echo "  Check log: ${LOGFILE}"
    echo "=================================================================="
    exit ${EXIT_CODE}
fi
