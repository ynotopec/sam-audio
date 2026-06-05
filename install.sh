#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${SCRIPT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_BIN="${UV_BIN:-uv}"

# Official SAM-Audio dependency packages that are installed without transitive
# dependency resolution. This avoids the unconditional perception_models ->
# decord dependency, because decord has no Linux aarch64 wheels for CPython 3.11
# on DGX Spark / Grace systems. The compatible runtime dependencies needed by
# this app are listed explicitly in requirements.txt instead.
SAM_AUDIO_GIT_DEPS=(
  "git+https://github.com/facebookresearch/dacvae.git"
  "git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f"
  "git+https://github.com/facebookresearch/ImageBind.git"
  "git+https://github.com/lematt1991/CLAP.git"
  "git+https://github.com/facebookresearch/perception_models@unpin-deps"
  "git+https://github.com/facebookresearch/sam-audio.git"
)

cd "${SCRIPT_DIR}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${VENV_DIR}")"
"${UV_BIN}" venv --allow-existing --python "${PYTHON_VERSION}" "${VENV_DIR}"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade -r requirements.txt
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade --no-deps "${SAM_AUDIO_GIT_DEPS[@]}"

cat <<EOF
Installed/updated ${PROJECT_NAME} in ${VENV_DIR}
Run with: source run.sh 0.0.0.0 7860
EOF
