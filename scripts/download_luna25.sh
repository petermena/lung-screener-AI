#!/usr/bin/env bash
# Download LUNA25 dataset from Zenodo
# Run this on a machine with unrestricted internet (e.g., RunPod)
#
# Usage:
#   ./scripts/download_luna25.sh                    # Download everything
#   ./scripts/download_luna25.sh --nodules-only     # Download only nodule blocks (~6GB)
#   ./scripts/download_luna25.sh --images-only      # Download only CT images (~230GB)
#
# All files are in Zenodo record 14223624:
#   https://zenodo.org/records/14223624
#   - 46 image parts:  luna25_images.zip.001 .. .046
#   - 2 nodule parts:  luna25_nodule_blocks.zip.001 .. .002
#
# Annotation metadata is in Zenodo record 14673658:
#   https://zenodo.org/records/14673658
#
# On RunPod, data is stored on the persistent volume (/workspace) so it
# survives pod stop/restart. The script auto-detects RunPod and uses
# /workspace/luna25 as the data directory, with a symlink at ./data/luna25.

set -uo pipefail

# ---------- configuration ----------
ZENODO_IMAGES="https://zenodo.org/api/records/14223624/files"
ZENODO_ANNOTATIONS="https://zenodo.org/api/records/14673658/files"

IMAGE_PARTS=46        # luna25_images.zip.001 .. .046
NODULE_PARTS=2        # luna25_nodule_blocks.zip.001 .. .002

# ---------- parse arguments ----------
DOWNLOAD_IMAGES=true
DOWNLOAD_NODULES=true
DOWNLOAD_ANNOTATIONS=true

for arg in "$@"; do
    case "$arg" in
        --nodules-only)
            DOWNLOAD_IMAGES=false
            ;;
        --images-only)
            DOWNLOAD_NODULES=false
            ;;
        --no-annotations)
            DOWNLOAD_ANNOTATIONS=false
            ;;
        -h|--help)
            head -n 17 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--nodules-only|--images-only|--no-annotations]"
            exit 1
            ;;
    esac
done

# ---------- data directory ----------
if [ -d "/workspace" ]; then
    DATA_DIR="/workspace/luna25"
    LINK_DIR="./data/luna25"
    echo "=== RunPod detected: storing data on persistent volume ==="
    echo "    Data dir:  $DATA_DIR"
    echo "    Symlink:   $LINK_DIR -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$(dirname "$LINK_DIR")"
    if [ ! -L "$LINK_DIR" ] && [ ! -d "$LINK_DIR" ]; then
        ln -s "$DATA_DIR" "$LINK_DIR"
    elif [ -d "$LINK_DIR" ] && [ ! -L "$LINK_DIR" ]; then
        echo "    WARNING: ./data/luna25 is a real directory, not a symlink."
        echo "    Data will be saved to /workspace/luna25."
        echo "    You may want to: rm -rf ./data/luna25 && ln -s /workspace/luna25 ./data/luna25"
    fi
else
    DATA_DIR="./data/luna25"
fi
mkdir -p "$DATA_DIR"

# ---------- helpers ----------
download_file() {
    local url="$1"
    local output="$2"
    local max_retries=5

    if [ -f "$output" ]; then
        echo "  Already exists: $(basename "$output") (skipping)"
        return 0
    fi

    echo "  Downloading: $(basename "$output")"
    for attempt in $(seq 1 $max_retries); do
        if curl -L --progress-bar \
                -C - \
                --retry 3 --retry-delay 5 --retry-max-time 120 \
                --connect-timeout 30 \
                --fail \
                -o "$output" "$url"; then
            local file_size
            file_size=$(stat -c%s "$output" 2>/dev/null || echo "0")
            echo "  Downloaded $(basename "$output"): ${file_size} bytes"
            return 0
        else
            echo "  curl failed (exit code $?)"
        fi

        if [ "$attempt" -lt "$max_retries" ]; then
            local wait_secs=$((2 ** attempt))
            echo "  Retry $((attempt + 1))/$max_retries in ${wait_secs}s..."
            sleep "$wait_secs"
        fi
    done

    echo "  FAILED: $output after $max_retries attempts"
    rm -f "$output"
    return 1
}

# ---------- download annotation metadata ----------
if [ "$DOWNLOAD_ANNOTATIONS" = true ]; then
    echo ""
    echo "=== Downloading annotation metadata (record 14673658) ==="
    # Discover actual filenames from the Zenodo API
    ANNOT_FILES=""
    ANNOT_JSON=$(curl -sL "https://zenodo.org/api/records/14673658" 2>/dev/null || true)
    if [ -n "$ANNOT_JSON" ]; then
        if command -v jq &>/dev/null; then
            ANNOT_FILES=$(echo "$ANNOT_JSON" | jq -r '.files[]?.key // empty' 2>/dev/null || true)
        elif command -v python3 &>/dev/null; then
            ANNOT_FILES=$(echo "$ANNOT_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for f in data.get('files', []):
        print(f['key'])
except: pass
" 2>/dev/null || true)
        fi
    fi
    if [ -z "$ANNOT_FILES" ]; then
        echo "  WARNING: Could not discover annotation files from Zenodo API."
        echo "  Skipping annotations. Download them manually from:"
        echo "    https://zenodo.org/records/14673658"
    else
        echo "  Found annotation files: $ANNOT_FILES"
        for fname in $ANNOT_FILES; do
            download_file "${ZENODO_ANNOTATIONS}/${fname}/content" "$DATA_DIR/$fname" || true
        done
    fi
fi

# ---------- download nodule blocks ----------
if [ "$DOWNLOAD_NODULES" = true ]; then
    echo ""
    echo "=== Downloading nodule blocks (${NODULE_PARTS} parts) ==="
    mkdir -p "$DATA_DIR/nodule_blocks"
    for i in $(seq -w 1 "$NODULE_PARTS"); do
        # seq -w pads to width of the end value (2 digits: 01, 02)
        # but Zenodo files use 3-digit padding: .001, .002
        part=$(printf "%03d" "$((10#$i))")
        download_file \
            "${ZENODO_IMAGES}/luna25_nodule_blocks.zip.${part}/content" \
            "$DATA_DIR/nodule_blocks/luna25_nodule_blocks.zip.${part}"
    done

    echo ""
    echo "  To reassemble and extract nodule blocks:"
    echo "    cd $DATA_DIR/nodule_blocks"
    echo "    cat luna25_nodule_blocks.zip.* > luna25_nodule_blocks.zip"
    echo "    unzip luna25_nodule_blocks.zip"
fi

# ---------- download CT images ----------
if [ "$DOWNLOAD_IMAGES" = true ]; then
    echo ""
    echo "=== Downloading CT images (${IMAGE_PARTS} parts, ~230GB total) ==="
    mkdir -p "$DATA_DIR/images"
    for i in $(seq -w 1 "$IMAGE_PARTS"); do
        part=$(printf "%03d" "$((10#$i))")
        download_file \
            "${ZENODO_IMAGES}/luna25_images.zip.${part}/content" \
            "$DATA_DIR/images/luna25_images.zip.${part}"
    done

    echo ""
    echo "  To reassemble and extract images:"
    echo "    cd $DATA_DIR/images"
    echo "    cat luna25_images.zip.* > luna25_images.zip"
    echo "    unzip luna25_images.zip"
fi

# ---------- summary ----------
echo ""
echo "=== Download complete ==="
echo ""
echo "Dataset structure:"
ls -la "$DATA_DIR/"
echo ""
echo "To get started quickly with just nodule blocks:"
echo "  ./scripts/download_luna25.sh --nodules-only"
echo ""
echo "Ready to train:"
echo "  lung-screener train --checkpoint-dir ./checkpoints"
