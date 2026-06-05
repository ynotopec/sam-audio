# SAM-Audio minimal server

SAM-Audio Gradio UI plus a token-protected FastAPI REST API.

## Install / compatible upgrade

```bash
./install.sh
```

The installer is idempotent and uses `uv` to create/update the virtual environment at:

```text
~/venv/sam-audio
```

SAM-Audio and its official GitHub-only components are installed by `install.sh` after the PyPI dependencies. They are installed with explicit runtime dependencies so Linux aarch64 systems such as DGX Spark do not fail on the optional `decord` video wheel dependency. The explicit runtime dependencies include ImageBind's PyTorchVideo stack (`fvcore`, `av`, `parameterized`, and `networkx`) so SAM-Audio's visual ranker can be imported when the model initializes. The installer finishes with an import check for ImageBind and SAM-Audio, which fails early if a runtime dependency is still missing. The app also includes an import-time `xformers.ops` fallback for the default PyTorch SDPA inference path, because xformers wheels are not consistently available on Linux aarch64.

You can override the Python version or environment path:

```bash
PYTHON_VERSION=3.11 VENV_DIR=~/venv/sam-audio ./install.sh
```

## Configure

```bash
cp .env.example .env
nano .env
```

Set `API_TOKEN` to require token auth for API calls. Comment it out or leave it empty only on trusted private networks.

## Start

```bash
source run.sh 0.0.0.0 7860
```

Open the UI at `http://127.0.0.1:7860/ui`.

`run.sh` keeps the server in the foreground, so it is compatible with systemd `ExecStart`:

```ini
[Service]
WorkingDirectory=/workspace/sam-audio
ExecStart=/bin/bash -lc 'source /workspace/sam-audio/run.sh 0.0.0.0 7860'
Restart=always
```

## REST API

Health check:

```bash
curl http://127.0.0.1:7860/health
```

Separate audio:

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/separations \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -F "file=@input.wav" \
  -F 'description=A man speaking' \
  -F 'anchors=' \
  -F 'reranking_candidates=0' \
  -F 'predict_spans=true' \
  -o response.json
```

The response contains base64-encoded WAV files for `target` and `residual`.

## GPU notes

The default `SAM_AUDIO_DTYPE=auto` uses BF16 on CUDA GPUs that support it, which is appropriate for H100/DGX-class systems, and falls back to FP16 otherwise.
