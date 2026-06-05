#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_BIN="${UV_BIN:-uv}"

cd "${SCRIPT_DIR}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${VENV_DIR}")"
"${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade -r requirements.txt

cat <<EOF
Installed/updated ${PROJECT_NAME} in ${VENV_DIR}
Run with: source run.sh 0.0.0.0 7860
EOF
