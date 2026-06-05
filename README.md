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
