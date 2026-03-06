#!/usr/bin/env python3
"""Package Lung Screener AI for offline deployment.

Creates a self-contained distribution that can be installed on any
machine without internet access. Two modes:

1. ONNX mode (default): ~200MB total, no PyTorch needed on target
2. Full mode: ~2.5GB, includes PyTorch for GPU inference

Usage:
    # After training, create offline package with ONNX model:
    python scripts/package_offline.py --checkpoint best.pth --output dist/

    # Full PyTorch package (larger but supports GPU):
    python scripts/package_offline.py --checkpoint best.pth --output dist/ --full

    # On the target machine:
    cd dist/lung-screener-offline/
    ./install.sh          # or: pip install --no-index --find-links wheels/ -r requirements.txt
    lung-screener predict /path/to/dicom -m model/model.onnx
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Viewer (web UI) dependencies — always included
VIEWER_REQUIREMENTS = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "python-multipart>=0.0.6",
]

# Minimal dependencies for ONNX-based offline inference
ONNX_REQUIREMENTS = [
    "onnxruntime>=1.16",
    "pydicom>=2.4",
    "pynetdicom>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.21",
    "SimpleITK>=2.3",
    "pyyaml>=6.0",
    "click>=8.1",
] + VIEWER_REQUIREMENTS

# Full requirements (includes PyTorch)
FULL_REQUIREMENTS = [
    "torch>=2.0",
    "pydicom>=2.4",
    "pynetdicom>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.21",
    "SimpleITK>=2.3",
    "pyyaml>=6.0",
    "click>=8.1",
] + VIEWER_REQUIREMENTS


def download_wheels(requirements: list[str], dest: Path, platform: str | None = None, python_version: str = "310"):
    """Download wheel files for all dependencies."""
    dest.mkdir(parents=True, exist_ok=True)

    # Always download build tools so lung_screener_pkg can be installed offline
    build_tools = ["setuptools>=68.0", "wheel>=0.40", "pip>=23.0"]

    def _download(pkgs: list[str], extra_flags: list[str] | None = None):
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(dest),
            "--only-binary", ":all:",
        ]
        if platform:
            cmd.extend(["--platform", platform, "--python-version", python_version])
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend(pkgs)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Some wheels could not be fetched as binaries:\n{result.stderr}")
            # Retry without --only-binary (allows sdists for pure-Python packages)
            cmd_fallback = [sys.executable, "-m", "pip", "download", "--dest", str(dest)]
            if platform:
                cmd_fallback.extend(["--platform", platform, "--python-version", python_version])
            cmd_fallback.extend(pkgs)
            subprocess.run(cmd_fallback, check=True)

    logger.info(f"Downloading wheels to {dest}...")
    _download(requirements)
    logger.info("Downloading build tools (setuptools, wheel, pip)...")
    _download(build_tools)


def export_onnx_model(checkpoint: Path, output: Path, project_root: Path):
    """Export the trained model to ONNX format."""
    sys.path.insert(0, str(project_root))
    import yaml

    from lung_screener.export import export_to_onnx

    config_path = project_root / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    export_to_onnx(checkpoint, output, config)


def create_install_script(package_dir: Path, mode: str):
    """Create the install.sh script for the target machine."""
    script = f"""#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
echo "=== Lung Screener AI - Offline Installer ==="
echo ""

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'" 2>/dev/null || {{
    echo "ERROR: Python 3.10 or later is required."
    echo "Install it from https://www.python.org/downloads/"
    exit 1
}}

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$SCRIPT_DIR/venv"
source "$SCRIPT_DIR/venv/bin/activate"

# Install from local wheels (no internet needed)
echo "Installing dependencies from bundled wheels..."
pip install --upgrade pip setuptools wheel --no-index --find-links "$SCRIPT_DIR/wheels"
pip install --no-index --find-links "$SCRIPT_DIR/wheels" -r "$SCRIPT_DIR/requirements.txt"

# Install the lung_screener package itself
pip install --no-index --find-links "$SCRIPT_DIR/wheels" lung-screener-ai

echo ""
echo "=== Installation complete ==="
echo ""
echo "To use Lung Screener AI:"
echo "  source $SCRIPT_DIR/venv/bin/activate"
echo "  lung-screener predict /path/to/dicom/series -m $SCRIPT_DIR/model/model.{'onnx' if mode == 'onnx' else 'pth'}"
echo ""
echo "To start the web viewer (fully offline, open http://localhost:8080):"
echo "  lung-screener viewer -m $SCRIPT_DIR/model/model.{'onnx' if mode == 'onnx' else 'pth'}"
echo ""
echo "To start PACS listener:"
echo "  lung-screener serve -m $SCRIPT_DIR/model/model.{'onnx' if mode == 'onnx' else 'pth'}"
"""
    install_path = package_dir / "install.sh"
    install_path.write_text(script)
    install_path.chmod(0o755)

    # Windows batch file
    bat_script = f"""@echo off
echo === Lung Screener AI - Offline Installer ===
echo.

python -c "import sys; assert sys.version_info >= (3, 10)" 2>NUL
if errorlevel 1 (
    echo ERROR: Python 3.10 or later is required.
    exit /b 1
)

echo Creating virtual environment...
python -m venv "%~dp0venv"
call "%~dp0venv\\Scripts\\activate.bat"

echo Installing dependencies from bundled wheels...
pip install --upgrade pip setuptools wheel --no-index --find-links "%~dp0wheels"
pip install --no-index --find-links "%~dp0wheels" -r "%~dp0requirements.txt"
pip install --no-index --find-links "%~dp0wheels" lung-screener-ai

echo.
echo === Installation complete ===
echo.
echo Activate with: %~dp0venv\\Scripts\\activate.bat
echo Then run:      lung-screener predict C:\\path\\to\\dicom -m %~dp0model\\model.{'onnx' if mode == 'onnx' else 'pth'}
echo Or viewer:     lung-screener viewer -m %~dp0model\\model.{'onnx' if mode == 'onnx' else 'pth'}
"""
    bat_path = package_dir / "install.bat"
    bat_path.write_text(bat_script)


def main():
    parser = argparse.ArgumentParser(description="Package Lung Screener AI for offline use")
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output", default="./dist", type=Path,
        help="Output directory for the package",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include full PyTorch (larger package, GPU support)",
    )
    parser.add_argument(
        "--onnx-gpu", action="store_true",
        help="ONNX mode: use onnxruntime-gpu instead of onnxruntime (CUDA GPU support)",
    )
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Target platform for wheels (e.g., manylinux2014_x86_64, win_amd64, macosx_11_0_x86_64)",
    )
    parser.add_argument(
        "--python-version", type=str, default="310",
        help="Target Python version for wheels (e.g., 310, 311, 312, 314). Default: 310",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    mode = "full" if args.full else "onnx"

    package_dir = args.output / "lung-screener-offline"
    if package_dir.exists():
        shutil.rmtree(package_dir)

    # Create directory structure
    (package_dir / "model").mkdir(parents=True)
    (package_dir / "wheels").mkdir(parents=True)
    (package_dir / "lung_screener_pkg").mkdir(parents=True)

    # 1. Export or copy model
    logger.info("--- Step 1: Preparing model ---")
    if mode == "onnx":
        export_onnx_model(
            args.checkpoint,
            package_dir / "model" / "model.onnx",
            project_root,
        )
        requirements = ONNX_REQUIREMENTS.copy()
        if args.onnx_gpu:
            # Replace CPU-only onnxruntime with the GPU-enabled build
            requirements = [
                "onnxruntime-gpu>=1.16" if r == "onnxruntime>=1.16" else r
                for r in requirements
            ]
            logger.info("ONNX GPU mode: using onnxruntime-gpu")
    else:
        shutil.copy2(args.checkpoint, package_dir / "model" / "model.pth")
        requirements = FULL_REQUIREMENTS

    # 2. Download dependency wheels
    logger.info("--- Step 2: Downloading dependency wheels ---")
    download_wheels(requirements, package_dir / "wheels", args.platform, args.python_version)

    # 3. Build lung_screener as a wheel and place it in the wheels directory
    logger.info("--- Step 3: Building lung_screener wheel ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", tmp, str(project_root)],
            check=True,
        )
        for whl in Path(tmp).glob("*.whl"):
            shutil.copy2(whl, package_dir / "wheels" / whl.name)
            logger.info(f"  Built: {whl.name}")

    # 4. Copy config (still needed at runtime for model defaults)
    logger.info("--- Step 4: Copying configuration ---")
    config_dest = package_dir / "lung_screener_pkg" / "config"
    shutil.copytree(project_root / "config", config_dest)

    # 5. Write requirements.txt
    req_path = package_dir / "requirements.txt"
    req_path.write_text("\n".join(requirements) + "\n")

    # 6. Create install scripts
    logger.info("--- Step 5: Creating install scripts ---")
    create_install_script(package_dir, mode)

    # 7. Create archive
    logger.info("--- Step 6: Creating archive ---")
    archive_path = shutil.make_archive(
        str(args.output / "lung-screener-offline"),
        "zip",
        root_dir=str(args.output),
        base_dir="lung-screener-offline",
    )

    total_size = sum(f.stat().st_size for f in package_dir.rglob("*") if f.is_file())
    logger.info("")
    logger.info("=== Package created successfully ===")
    logger.info(f"  Directory: {package_dir}")
    logger.info(f"  Archive:   {archive_path}")
    logger.info(f"  Size:      {total_size / 1024 / 1024:.0f} MB")
    logger.info(f"  Mode:      {mode} ({'lightweight, CPU' if mode == 'onnx' else 'full, GPU support'})")
    logger.info("")
    logger.info("To deploy on a target machine:")
    logger.info(f"  1. Copy {archive_path} to the target machine")
    logger.info("  2. Unzip it")
    logger.info("  3. Run: ./lung-screener-offline/install.sh")


if __name__ == "__main__":
    main()
