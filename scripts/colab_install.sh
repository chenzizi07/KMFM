#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-KMFM}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/Colab/Unsupervised}"
PROJECT_DIR="${PROJECT_DIR:-${DRIVE_ROOT}/${PROJECT_NAME}}"
REPO_URL="${REPO_URL:-https://github.com/chenzizi07/KMFM.git}"

mkdir -p "${DRIVE_ROOT}"
if [[ -d "${PROJECT_DIR}/.git" ]]; then
  bash "${PROJECT_DIR}/scripts/colab_update.sh"
  exit 0
fi
if [[ -e "${PROJECT_DIR}" && -n "$(find "${PROJECT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[install] refusing to overwrite non-git directory: ${PROJECT_DIR}"
  exit 2
fi

git clone -- "${REPO_URL}" "${PROJECT_DIR}"
python "${PROJECT_DIR}/scripts/colab_git.py" \
  --repo-url "${REPO_URL}" \
  --project-dir "${PROJECT_DIR}" \
  --mode update
