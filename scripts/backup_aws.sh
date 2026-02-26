#!/usr/bin/env bash
# backup_aws.sh — Safely back up all AWS training artifacts to GitHub and S3.
#
# Usage:
#   ./scripts/backup_aws.sh                         # Git only
#   ./scripts/backup_aws.sh --s3 s3://my-bucket/lung-screener-AI
#   ./scripts/backup_aws.sh --s3 s3://my-bucket/lung-screener-AI --msg "epoch 142 best checkpoint"
#
# What it does:
#   1. Stashes any uncommitted local changes
#   2. Pulls the latest remote branch (rebase)
#   3. Restores your stashed changes on top
#   4. Commits all tracked checkpoint metadata (eval_results.json, metrics.json,
#      dashboard.html, calibration.json) plus any other staged/modified tracked files
#   5. Pushes to GitHub
#   6. (Optional) Uploads model weights and ONNX exports to S3

set -euo pipefail

BRANCH="claude/resume-lung-nodule-system-usHuW"
CHECKPOINT_DIR="./checkpoints"
S3_DEST=""
COMMIT_MSG=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --s3)       S3_DEST="$2";    shift 2 ;;
        --msg|-m)   COMMIT_MSG="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " Lung Screener AI — AWS Backup"
echo " Branch : $BRANCH"
echo " S3     : ${S3_DEST:-'(skipped — pass --s3 to enable)'}"
echo "============================================================"

# ── Sanity check ──────────────────────────────────────────────────────────────
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not inside a git repository. Run from the repo root." >&2
    exit 1
fi

# ── Step 1: Stash local changes so checkout/pull can proceed ─────────────────
echo ""
echo "[1/5] Stashing local changes..."
STASH_RESULT=$(git stash 2>&1)
echo "$STASH_RESULT"
if echo "$STASH_RESULT" | grep -q "No local changes"; then
    STASHED=0
else
    STASHED=1
fi

# ── Step 2: Switch to target branch and pull ─────────────────────────────────
echo ""
echo "[2/5] Switching to $BRANCH and pulling latest..."
git checkout "$BRANCH"
git pull --rebase origin "$BRANCH"

# ── Step 3: Restore local changes on top ─────────────────────────────────────
if [[ "$STASHED" -eq 1 ]]; then
    echo ""
    echo "[3/5] Restoring stashed changes..."
    if ! git stash pop; then
        echo ""
        echo "CONFLICT detected. Keeping your local versions of conflicted files..."
        # For checkpoint metadata, always prefer the local (AWS) version
        for f in \
            "$CHECKPOINT_DIR/eval_results.json" \
            "$CHECKPOINT_DIR/metrics.json" \
            "$CHECKPOINT_DIR/dashboard.html" \
            "$CHECKPOINT_DIR/calibration.json"
        do
            if [[ -f "$f" ]]; then
                git checkout --theirs "$f" 2>/dev/null && git add "$f" && \
                    echo "  kept local: $f" || true
            fi
        done
        git stash drop 2>/dev/null || true
    fi
else
    echo ""
    echo "[3/5] No stash to restore."
fi

# ── Step 4: Commit all tracked checkpoint metadata ───────────────────────────
echo ""
echo "[4/5] Committing tracked checkpoint files..."

TRACKED_FILES=()
for f in \
    "$CHECKPOINT_DIR/eval_results.json" \
    "$CHECKPOINT_DIR/metrics.json" \
    "$CHECKPOINT_DIR/dashboard.html" \
    "$CHECKPOINT_DIR/calibration.json"
do
    if [[ -f "$f" ]]; then
        git add "$f"
        TRACKED_FILES+=("$f")
    fi
done

# Also stage any other modified tracked files (e.g. config, source changes)
git add -u

if git diff --cached --quiet; then
    echo "  Nothing new to commit — GitHub is already up to date."
else
    # Auto-generate commit message if not provided
    if [[ -z "$COMMIT_MSG" ]]; then
        EPOCH="unknown"
        if [[ -f "$CHECKPOINT_DIR/eval_results.json" ]]; then
            EPOCH=$(python3 -c "
import json, sys
try:
    d = json.load(open('$CHECKPOINT_DIR/eval_results.json'))
    auc = d.get('metrics', {}).get('auc_roc', '')
    ep  = d.get('epoch', '')
    print(f'epoch {ep}, AUC {auc}')
except Exception:
    print('updated')
" 2>/dev/null || echo "updated")
        fi
        COMMIT_MSG="Update checkpoint metadata: ${EPOCH}"
    fi

    git commit -m "$COMMIT_MSG"
    echo "  Committed: $COMMIT_MSG"
fi

# ── Step 5: Push to GitHub ────────────────────────────────────────────────────
echo ""
echo "[5/5] Pushing to GitHub..."
PUSH_ATTEMPT=0
PUSH_DELAYS=(2 4 8 16)
until git push -u origin "$BRANCH"; do
    PUSH_ATTEMPT=$((PUSH_ATTEMPT + 1))
    if [[ $PUSH_ATTEMPT -gt ${#PUSH_DELAYS[@]} ]]; then
        echo "ERROR: Push failed after ${#PUSH_DELAYS[@]} retries." >&2
        exit 1
    fi
    DELAY=${PUSH_DELAYS[$((PUSH_ATTEMPT - 1))]}
    echo "  Push failed. Retrying in ${DELAY}s... (attempt $PUSH_ATTEMPT/${#PUSH_DELAYS[@]})"
    sleep "$DELAY"
done
echo "  Pushed to $BRANCH"

# ── Optional: S3 backup for model weights and ONNX exports ───────────────────
if [[ -n "$S3_DEST" ]]; then
    echo ""
    echo "[S3] Uploading model artifacts to $S3_DEST ..."

    if ! command -v aws &>/dev/null; then
        echo "  WARNING: aws CLI not found. Install with: pip install awscli" >&2
    else
        UPLOADED=0
        for f in \
            "$CHECKPOINT_DIR/best.pth" \
            "$CHECKPOINT_DIR"/*.pth \
            "$CHECKPOINT_DIR"/*.onnx \
            "$CHECKPOINT_DIR/calibration.json"
        do
            # Expand glob — skip if no matches
            [[ -e "$f" ]] || continue
            DEST_KEY="$S3_DEST/$(basename "$f")"
            echo "  Uploading $(basename "$f") ..."
            aws s3 cp "$f" "$DEST_KEY" --no-progress
            UPLOADED=$((UPLOADED + 1))
        done
        echo "  $UPLOADED file(s) uploaded to S3."
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Backup complete."
echo ""
echo " GitHub (tracked metadata):"
for f in "${TRACKED_FILES[@]}"; do
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "   $f  ($SIZE)"
done
echo ""
echo " NOT in GitHub (gitignored — back up to S3 separately):"
for f in \
    "$CHECKPOINT_DIR/best.pth" \
    "$CHECKPOINT_DIR"/*.onnx
do
    [[ -e "$f" ]] || continue
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "   $f  ($SIZE)  <-- run with --s3 s3://your-bucket/path to upload"
done
echo "============================================================"
