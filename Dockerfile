# Lung Screener AI - Offline Docker Image (ONNX Runtime)
#
# Lightweight deployment image using ONNX Runtime (~50MB model footprint).
# PyTorch is used only during the build stage to export the model, then removed.
#
# Build (after training):
#   docker build -t lung-screener --build-arg MODEL_PATH=./checkpoints/best.pth .
#
# Or use a pre-exported ONNX model:
#   docker build -t lung-screener --build-arg ONNX_PATH=./checkpoints/model.onnx -f Dockerfile .
#
# Run inference on a DICOM directory:
#   docker run --rm -v /path/to/dicoms:/data lung-screener predict /data -m /app/model/model.onnx
#
# Run as PACS listener:
#   docker run -d -p 11112:11112 --name lung-screener lung-screener serve -m /app/model/model.onnx
#
# ========================================================================
# Stage 1: Export PyTorch checkpoint to ONNX (builder stage)
# ========================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install minimal PyTorch for export only
RUN pip install --no-cache-dir torch>=2.0 pyyaml>=6.0

COPY lung_screener/ lung_screener/
COPY config/ config/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

ARG MODEL_PATH=./checkpoints/best.pth
ARG ONNX_PATH=""

# Export if .pth provided, or copy if .onnx provided
COPY ${MODEL_PATH} /build/checkpoint.pth
RUN mkdir -p /build/model && \
    if [ -n "${ONNX_PATH}" ]; then \
        cp /build/checkpoint.pth /build/model/model.onnx; \
    else \
        python -c " \
import yaml; \
from lung_screener.export import export_to_onnx; \
config = yaml.safe_load(open('config/default.yaml')); \
export_to_onnx('checkpoint.pth', '/build/model/model.onnx', config); \
"; \
    fi

# ========================================================================
# Stage 2: Lightweight runtime image (no PyTorch)
# ========================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# System dependencies for SimpleITK and scikit-image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (ONNX mode only - lightweight)
RUN pip install --no-cache-dir \
    onnxruntime>=1.16 \
    pydicom>=2.4 \
    pynetdicom>=2.0 \
    numpy>=1.24 \
    scipy>=1.10 \
    scikit-image>=0.21 \
    SimpleITK>=2.3 \
    pyyaml>=6.0 \
    click>=8.1

# Copy application code
COPY lung_screener/ lung_screener/
COPY config/ config/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Copy the ONNX model from builder stage
COPY --from=builder /build/model/model.onnx /app/model/model.onnx

# Copy calibration (defaults to checkpoints/calibration.json if it exists)
ARG CAL_PATH=checkpoints/calibration.json
COPY ${CAL_PATH} /app/model/calibration.json

# Expose DICOM SCP port
EXPOSE 11112

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import onnxruntime; sess = onnxruntime.InferenceSession('/app/model/model.onnx', providers=['CPUExecutionProvider']); print('ok')" || exit 1

ENTRYPOINT ["lung-screener"]
CMD ["--help"]
