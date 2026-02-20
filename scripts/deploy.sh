#!/usr/bin/env bash
# Lung Screener AI — Build and Deploy
#
# Usage:
#   ./scripts/deploy.sh              # Build ONNX image and start
#   ./scripts/deploy.sh --gpu        # Build GPU image and start
#   ./scripts/deploy.sh --build-only # Build without starting
#
# Prerequisites:
#   - Docker (and docker compose)
#   - Trained model at checkpoints/best.pth
#   - (GPU mode) NVIDIA Container Toolkit

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=false
BUILD_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --gpu)     GPU=true ;;
        --build-only) BUILD_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--gpu] [--build-only]"
            exit 0
            ;;
    esac
done

# Verify checkpoint exists
CHECKPOINT="./checkpoints/best.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: No trained model found at $CHECKPOINT"
    echo "Train a model first: lung-screener train"
    exit 1
fi

echo "============================================"
echo "  Lung Screener AI — Deploy"
echo "============================================"
echo ""

if [ "$GPU" = true ]; then
    echo "Mode: GPU (PyTorch + CUDA)"
    echo "Building lung-screener-gpu image..."
    docker compose build lung-screener-gpu

    if [ "$BUILD_ONLY" = true ]; then
        echo "Build complete. Start with: docker compose --profile gpu up -d"
        exit 0
    fi

    echo "Starting DICOM SCP server (GPU)..."
    docker compose --profile gpu up -d
    echo ""
    echo "Service started. DICOM SCP listening on port 11112."
    echo ""
    echo "Commands:"
    echo "  docker compose logs -f lung-screener-gpu   # View logs"
    echo "  docker compose --profile gpu down           # Stop"
else
    echo "Mode: ONNX (CPU, lightweight)"
    echo "Building lung-screener image..."
    docker compose build lung-screener

    if [ "$BUILD_ONLY" = true ]; then
        echo "Build complete. Start with: docker compose up -d"
        exit 0
    fi

    echo "Starting DICOM SCP server..."
    docker compose up -d lung-screener
    echo ""
    echo "Service started. DICOM SCP listening on port 11112."
    echo ""
    echo "Commands:"
    echo "  docker compose logs -f lung-screener   # View logs"
    echo "  docker compose down                    # Stop"
fi

echo ""
echo "Send a CT study with:"
echo "  storescu localhost 11112 -aec LUNG_SCREEN_AI /path/to/*.dcm"
