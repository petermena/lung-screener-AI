#!/usr/bin/env bash
# =============================================================================
# Step 3: Docker Deploy — build ONNX runtime image, expose DICOM SCP
#
# What this script does:
#   1. Pre-exports checkpoints/best.pth → checkpoints/model.onnx (optional
#      but faster than letting the Dockerfile do it inside the build).
#   2. Builds the lung-screener:latest Docker image (CPU / ONNX runtime).
#   3. Starts the container with docker compose (DICOM SCP on port 11112).
#   4. Runs a quick smoke-test to confirm the model loads correctly.
#
# Prerequisites:
#   - Docker >= 24 with BuildKit
#   - checkpoints/best.pth  (first-stage trained checkpoint)
#   - Python environment with lung-screener installed (for ONNX export)
#
# Estimated time: ~5-10 minutes (ONNX export ~1 min, docker build ~5-8 min)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/docker_deploy_$(date +%Y%m%d_%H%M%S).log"

CKPT="checkpoints/best.pth"
ONNX_OUT="checkpoints/model.onnx"
IMAGE_TAG="lung-screener:latest"

echo "=================================================================="
echo "  DOCKER DEPLOY — $(date)"
echo "=================================================================="
echo "  Project:    ${PROJECT_ROOT}"
echo "  Checkpoint: ${CKPT}"
echo "  ONNX out:   ${ONNX_OUT}"
echo "  Image:      ${IMAGE_TAG}"
echo "  Log file:   ${LOGFILE}"
echo "=================================================================="

# ------------------------------------------------------------------
# 1. Verify checkpoint
# ------------------------------------------------------------------
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found. Run FP reduction or main training first."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Export PyTorch checkpoint → ONNX (on host, before docker build)
#    This avoids installing PyTorch inside the builder layer every time.
# ------------------------------------------------------------------
if [[ ! -f "${ONNX_OUT}" ]]; then
    echo ""
    echo "[$(date +%H:%M:%S)] Exporting ${CKPT} to ONNX..."
    python3 - <<'PYEOF' 2>&1 | tee -a "${LOGFILE}"
import yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent if "__file__" in dir() else "."))
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(".").resolve()))
from lung_screener.export import export_to_onnx
config = yaml.safe_load(open("config/default.yaml"))
out = export_to_onnx("checkpoints/best.pth", "checkpoints/model.onnx", config)
print(f"  ONNX model saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
PYEOF
else
    echo "[$(date +%H:%M:%S)] ONNX model already exists at ${ONNX_OUT}, skipping export."
fi

# ------------------------------------------------------------------
# 3. Handle optional calibration file
# ------------------------------------------------------------------
CAL_FILE="checkpoints/calibration.json"
CAL_BUILD_ARG=""
if [[ -f "${CAL_FILE}" ]]; then
    echo "[$(date +%H:%M:%S)] Calibration file found: ${CAL_FILE}"
    CAL_BUILD_ARG="--build-arg CAL_PATH=${CAL_FILE}"
else
    # Create a minimal stub so the Dockerfile COPY doesn't fail
    echo '[{"note": "no calibration applied"}]' > "${CAL_FILE}"
    echo "[$(date +%H:%M:%S)] No calibration.json found — created stub at ${CAL_FILE}"
fi

# ------------------------------------------------------------------
# 4. Build Docker image
# ------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] Building Docker image: ${IMAGE_TAG} ..."
DOCKER_BUILDKIT=1 docker build \
    --build-arg MODEL_PATH="${CKPT}" \
    --build-arg ONNX_PATH="${ONNX_OUT}" \
    ${CAL_BUILD_ARG} \
    -t "${IMAGE_TAG}" \
    -f Dockerfile \
    . \
    2>&1 | tee -a "${LOGFILE}"

echo ""
echo "[$(date +%H:%M:%S)] Image built successfully."
docker images "${IMAGE_TAG}"

# ------------------------------------------------------------------
# 5. Smoke-test: verify ONNX model loads inside the container
# ------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] Running smoke-test (ONNX model load)..."
docker run --rm "${IMAGE_TAG}" python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('/app/model/model.onnx', providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
print(f'  ONNX model OK — input: {inp.name} {inp.shape}')
" 2>&1 | tee -a "${LOGFILE}"

# ------------------------------------------------------------------
# 6. Start DICOM SCP with docker compose
# ------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] Starting DICOM SCP (port 11112) via docker compose..."
mkdir -p data/incoming data/feedback logs

docker compose up -d lung-screener 2>&1 | tee -a "${LOGFILE}"

# Wait briefly then show status
sleep 5
docker compose ps

echo ""
echo "=================================================================="
echo "  DOCKER DEPLOY COMPLETE — $(date)"
echo ""
echo "  DICOM SCP is listening on port 11112"
echo "  AE Title:  LUNG_SCREEN_AI"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f lung-screener   # stream logs"
echo "    docker compose down                    # stop"
echo "    docker compose --profile gpu up -d     # GPU variant"
echo ""
echo "  Test DICOM echo:"
echo "    echoscu -aec LUNG_SCREEN_AI 127.0.0.1 11112"
echo ""
echo "  Full log: ${LOGFILE}"
echo "=================================================================="
