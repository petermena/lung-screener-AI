#!/usr/bin/env bash
# Download LUNA16 dataset from Zenodo
# Run this on a machine with unrestricted internet (e.g., RunPod)
#
# Usage:
#   ./scripts/download_luna16.sh              # Download all subsets (~100GB)
#   ./scripts/download_luna16.sh 0 1          # Download only subset0 and subset1
#
# On RunPod, data is stored on the persistent volume (/workspace) so it
# survives pod stop/restart. The script auto-detects RunPod and uses
# /workspace/luna16 as the data directory, with a symlink at ./data/luna16.
#
# Source: https://zenodo.org/records/3723295 (Part 1: subsets 0-6)
#         https://zenodo.org/records/4121926 (Part 2: subsets 7-9)
# Uses Zenodo API download format (post-Oct 2023 platform upgrade).

set -euo pipefail

# Auto-detect RunPod: use persistent volume so data survives pod restarts
if [ -d "/workspace" ]; then
    DATA_DIR="/workspace/luna16"
    LINK_DIR="./data/luna16"
    echo "=== RunPod detected: storing data on persistent volume ==="
    echo "    Data dir:  $DATA_DIR"
    echo "    Symlink:   $LINK_DIR -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$(dirname "$LINK_DIR")"
    # Create symlink so the project config (data.dataset_dir: ./data/luna16) works
    if [ ! -L "$LINK_DIR" ] && [ ! -d "$LINK_DIR" ]; then
        ln -s "$DATA_DIR" "$LINK_DIR"
    elif [ -d "$LINK_DIR" ] && [ ! -L "$LINK_DIR" ]; then
        echo "    WARNING: ./data/luna16 is a real directory, not a symlink."
        echo "    Data will be saved to /workspace/luna16."
        echo "    You may want to: rm -rf ./data/luna16 && ln -s /workspace/luna16 ./data/luna16"
    fi
else
    DATA_DIR="./data/luna16"
fi
mkdir -p "$DATA_DIR"

# Extract zip files using whichever tool is available
extract_zip() {
    local zip_file="$1"
    local dest_dir="$2"

    if command -v unzip &>/dev/null; then
        unzip -q -o "$zip_file" -d "$dest_dir"
    elif command -v python3 &>/dev/null; then
        python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip_file" "$dest_dir"
    elif command -v python &>/dev/null; then
        python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip_file" "$dest_dir"
    else
        echo "ERROR: No unzip tool found. Install with: apt-get install -y unzip"
        return 1
    fi
}

# Zenodo API base URLs (post-Oct 2023 platform upgrade)
# Old format /records/{ID}/files/{FILE}?download=1 returns HTML, not the file.
# New format /api/records/{ID}/files/{FILE}/content returns the actual file.
ZENODO_PART1="https://zenodo.org/api/records/3723295/files"
ZENODO_PART2="https://zenodo.org/api/records/4121926/files"

# Download a file with retry logic
is_valid_zip() {
    local file="$1"
    if command -v unzip &>/dev/null; then
        unzip -t "$file" &>/dev/null
    elif command -v python3 &>/dev/null; then
        python3 -c "import zipfile, sys; z=zipfile.ZipFile(sys.argv[1]); z.testzip(); z.close()" "$file" 2>/dev/null
    else
        # If we can't test, assume valid
        return 0
    fi
}

download_file() {
    local url="$1"
    local output="$2"
    local max_retries=5

    if [ -f "$output" ]; then
        # For zip files, verify integrity; remove if corrupt
        if [[ "$output" == *.zip ]]; then
            if is_valid_zip "$output"; then
                echo "  Already exists (verified): $output (skipping)"
                return 0
            else
                local bad_size
                bad_size=$(stat -c%s "$output" 2>/dev/null || echo "unknown")
                echo "  Corrupt zip detected (${bad_size} bytes), re-downloading: $output"
                rm -f "$output"
            fi
        else
            echo "  Already exists: $output (skipping)"
            return 0
        fi
    fi

    echo "  Downloading: $(basename "$output")"
    for attempt in $(seq 1 $max_retries); do
        # Use -C - to resume partial downloads, --retry for transient HTTP errors
        if curl -L --progress-bar \
                -C - \
                --retry 3 --retry-delay 5 --retry-max-time 120 \
                --connect-timeout 30 \
                -o "$output" "${url}/content"; then
            # Log file size for diagnostics
            local file_size
            file_size=$(stat -c%s "$output" 2>/dev/null || echo "0")
            echo "  Downloaded $(basename "$output"): ${file_size} bytes"

            # Subset zips are multi-GB; small files are HTML error pages
            if [[ "$output" == *.zip ]] && [ "$file_size" -lt 1000000 ]; then
                echo "  File too small (${file_size} bytes) — likely an error page, retrying..."
                rm -f "$output"
            elif [[ "$output" == *.zip ]] && ! is_valid_zip "$output"; then
                echo "  Downloaded file is corrupt (${file_size} bytes), retrying..."
                rm -f "$output"
            else
                return 0
            fi
        else
            echo "  curl failed (exit code $?)"
            # Don't remove partial file — next attempt will resume with -C -
        fi

        if [ "$attempt" -lt "$max_retries" ]; then
            local wait_secs=$((2 ** attempt))
            echo "  Retry $((attempt + 1))/$max_retries in ${wait_secs}s..."
            sleep "$wait_secs"
        fi
    done

    local final_size
    final_size=$(stat -c%s "$output" 2>/dev/null || echo "0")
    echo "  FAILED: $output (${final_size} bytes after $max_retries attempts)"
    rm -f "$output"  # Clean up failed download
    return 1
}

# --- Annotation files (small, always download) ---
echo "=== Downloading annotation files ==="
download_file "$ZENODO_PART1/annotations.csv" "$DATA_DIR/annotations.csv"
download_file "$ZENODO_PART1/candidates_V2.csv" "$DATA_DIR/candidates_V2.csv"
download_file "$ZENODO_PART1/sampleSubmission.csv" "$DATA_DIR/sampleSubmission.csv"

# --- CT scan subsets ---
# Determine which subsets to download
if [ $# -gt 0 ]; then
    SUBSETS=("$@")
    echo "=== Downloading selected subsets: ${SUBSETS[*]} ==="
else
    SUBSETS=(0 1 2 3 4 5 6 7 8 9)
    echo "=== Downloading all 10 subsets (~100GB total) ==="
fi

for i in "${SUBSETS[@]}"; do
    echo ""
    echo "--- Subset $i ---"

    # Subsets 0-6 are in Part 1, subsets 7-9 are in Part 2
    if [ "$i" -le 6 ]; then
        BASE_URL="$ZENODO_PART1"
    else
        BASE_URL="$ZENODO_PART2"
    fi

    ZIP_FILE="$DATA_DIR/subset${i}.zip"
    SUBSET_DIR="$DATA_DIR/subset${i}"

    # Download
    download_file "$BASE_URL/subset${i}.zip" "$ZIP_FILE"

    # Extract
    if [ -d "$SUBSET_DIR" ] && [ "$(ls -A "$SUBSET_DIR" 2>/dev/null)" ]; then
        echo "  Already extracted: $SUBSET_DIR (skipping)"
    else
        echo "  Extracting subset${i}.zip..."
        extract_zip "$ZIP_FILE" "$DATA_DIR"
        echo "  Done."
    fi

    # Remove zip to save space (optional — comment out to keep zips)
    if [ -d "$SUBSET_DIR" ] && [ "$(ls -A "$SUBSET_DIR" 2>/dev/null)" ]; then
        echo "  Removing zip to save disk space..."
        rm -f "$ZIP_FILE"
    fi
done

echo ""
echo "=== Download complete ==="
echo ""
echo "Dataset structure:"
ls -la "$DATA_DIR/"
echo ""
echo "Annotation stats:"
if [ -f "$DATA_DIR/annotations.csv" ]; then
    echo "  Annotations: $(wc -l < "$DATA_DIR/annotations.csv") lines"
fi
if [ -f "$DATA_DIR/candidates_V2.csv" ]; then
    echo "  Candidates:  $(wc -l < "$DATA_DIR/candidates_V2.csv") lines"
fi
echo ""
echo "Ready to train:"
echo "  lung-screener train --checkpoint-dir ./checkpoints"
