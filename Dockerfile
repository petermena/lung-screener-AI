# Lung Screener AI - Offline Docker Image
#
# Build (after training):
#   docker build -t lung-screener --build-arg MODEL_PATH=./checkpoints/best.pth .
#
# Run inference on a DICOM directory:
#   docker run --rm -v /path/to/dicoms:/data lung-screener predict /data -m /app/model/model.onnx
#
# Run as PACS listener:
#   docker run -d -p 11112:11112 --name lung-screener lung-screener serve -m /app/model/model.onnx
#
# With GPU support:
#   docker run --gpus all -d -p 11112:11112 lung-screener serve -m /app/model/model.onnx

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for SimpleITK and scikit-image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (ONNX mode - lightweight)
COPY pyproject.toml .
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

# Install the package
RUN pip install --no-cache-dir -e .

# Export model to ONNX during build (requires PyTorch temporarily)
ARG MODEL_PATH=./checkpoints/best.pth
COPY ${MODEL_PATH} /tmp/checkpoint.pth

RUN pip install --no-cache-dir torch>=2.0 && \
    python -c " \
import yaml, sys; \
sys.path.insert(0, '/app'); \
from lung_screener.export import export_to_onnx; \
config = yaml.safe_load(open('config/default.yaml')); \
export_to_onnx('/tmp/checkpoint.pth', '/app/model/model.onnx', config); \
" && \
    pip uninstall -y torch && \
    rm /tmp/checkpoint.pth

# Expose DICOM SCP port
EXPOSE 11112

ENTRYPOINT ["lung-screener"]
CMD ["--help"]
