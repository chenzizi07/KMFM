#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-KMFM}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/Colab/Unsupervised}"
PROJECT_DIR="${PROJECT_DIR:-${DRIVE_ROOT}/${PROJECT_NAME}}"
REPO_URL="${REPO_URL:-https://github.com/chenzizi07/KMFM.git}"

if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
  echo "[update] ${PROJECT_DIR} is not a Git repository. Run colab_install.sh first."
  exit 2
fi

python "${PROJECT_DIR}/scripts/colab_git.py" \
  --repo-url "${REPO_URL}" \
  --project-dir "${PROJECT_DIR}" \
  --mode update
echo "[update] done: ${PROJECT_DIR}"
