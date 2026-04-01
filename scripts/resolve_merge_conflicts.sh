#!/usr/bin/env bash
set -euo pipefail

TARGET_BRANCH="${1:-origin/main}"

# Ensure the custom merge driver exists locally.
git config merge.ours.driver true

echo "Fetching remote refs..."
git fetch --all --prune

echo "Merging ${TARGET_BRANCH} into $(git rev-parse --abbrev-ref HEAD)..."
if git merge "${TARGET_BRANCH}"; then
  echo "Merge completed without manual conflicts."
  exit 0
fi

echo "Auto-resolving known conflict files with branch versions..."
for f in .gitignore Dockerfile README.md docker-compose.yml pyproject.toml tests/test_model.py; do
  if git ls-files -u -- "$f" | grep -q .; then
    git checkout --ours -- "$f"
    git add "$f"
  fi
done

# If there are still unresolved conflicts, stop for manual resolution.
if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Unresolved conflicts remain:"
  git diff --name-only --diff-filter=U
  exit 2
fi

git commit -m "chore: resolve PR merge conflicts against ${TARGET_BRANCH}"
echo "Conflict resolution commit created."
