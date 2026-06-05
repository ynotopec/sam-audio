#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

set -a
if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source ".env"
fi
set +a

SERVER_HOST_ARG="${1:-${SERVER_HOST:-${GRADIO_SERVER_NAME:-0.0.0.0}}}"
SERVER_PORT_ARG="${2:-${PORT:-${SERVER_PORT:-7860}}}"
PROJECT_NAME="$(basename "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run ./install.sh first." >&2
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 1
  fi
  exit 1
fi

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export SERVER_HOST="${SERVER_HOST_ARG}"
export SERVER_NAME="${SERVER_HOST_ARG}"
export GRADIO_SERVER_NAME="${SERVER_HOST_ARG}"
export PORT="${SERVER_PORT_ARG}"
export SERVER_PORT="${SERVER_PORT_ARG}"
export GRADIO_SERVER_PORT="${SERVER_PORT_ARG}"
export BACK_PORT="${BACK_PORT:-$((SERVER_PORT_ARG + 1))}"

# Keep the process in the foreground for systemd, Docker, and shell usage.
exec "${PYTHON_BIN}" app.py --host "${SERVER_HOST_ARG}" --port "${SERVER_PORT_ARG}"
