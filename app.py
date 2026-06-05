import argparse
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import gradio as gr
import torch
import torchaudio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app
from huggingface_hub import login
from xformers_compat import ensure_xformers_ops

ensure_xformers_ops()

from sam_audio import SAMAudio, SAMAudioProcessor

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    login(token=_HF_TOKEN)

MODEL_ID = os.environ.get("SAM_AUDIO_MODEL", "facebook/sam-audio-small")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_NAME = os.environ.get("SAM_AUDIO_DTYPE", "auto").strip().lower()
API_TOKEN = os.environ.get("API_TOKEN", "").strip()

_model = None
_processor = None


def resolve_dtype():
    if DEVICE != "cuda":
        return None

    if DTYPE_NAME in {"fp16", "float16", "half"}:
        return torch.float16
    if DTYPE_NAME in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if DTYPE_NAME in {"fp32", "float32"}:
        return torch.float32
    # Auto prefers BF16 when available, which is the right default for H100-class GPUs.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_model():
    """Load the SAM-Audio model and processor once per process."""
    global _model, _processor
    if _model is None or _processor is None:
        dtype = resolve_dtype()
        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        _model = SAMAudio.from_pretrained(
            MODEL_ID,
            **model_kwargs,
        ).to(DEVICE).eval()

        _processor = SAMAudioProcessor.from_pretrained(
            MODEL_ID,
        )
    return _model, _processor


def parse_anchors(anchors_json: str):
    """
    Expected JSON shape:
    [
      ["+", 6.3, 7.0],
      ["-", 0.0, 2.0]
    ]
    Empty input disables anchors.
    """
    anchors_json = (anchors_json or "").strip()
    if not anchors_json:
        return None
    anchors = json.loads(anchors_json)
    # The processor expects anchors=[anchors] for one audio file.
    return [anchors]


def _separate_to_files(audio_path, description, anchors_json, reranking_candidates, predict_spans):
    description = (description or "").strip()
    if not description:
        raise ValueError("Description is required.")

    model, processor = get_model()

    anchors = parse_anchors(anchors_json)
    inputs = processor(
        audios=[audio_path],
        descriptions=[description],
        anchors=anchors,
    ).to(DEVICE)

    kwargs = {}
    if reranking_candidates and int(reranking_candidates) > 0:
        kwargs["reranking_candidates"] = int(reranking_candidates)

    try:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=resolve_dtype(), enabled=(DEVICE == "cuda")
        ):
            result = model.separate(inputs, predict_spans=bool(predict_spans), **kwargs)
    except torch.OutOfMemoryError as err:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OutOfMemory: try a smaller model, disable predict_spans, keep "
            "reranking_candidates at 0, or set SAM_AUDIO_DTYPE=bf16/fp16."
        ) from err

    sampling_rate = processor.audio_sampling_rate
    tmpdir = tempfile.mkdtemp(prefix="sam-audio-")
    target_path = os.path.join(tmpdir, "target.wav")
    residual_path = os.path.join(tmpdir, "residual.wav")

    torchaudio.save(target_path, result.target[0].unsqueeze(0).cpu(), sampling_rate)
    torchaudio.save(residual_path, result.residual[0].unsqueeze(0).cpu(), sampling_rate)

    spans = None
    if hasattr(result, "spans") and result.spans is not None:
        spans = result.spans

    return target_path, residual_path, spans, sampling_rate


def separate(audio_file, description, anchors_json, reranking_candidates, predict_spans):
    if audio_file is None:
        raise gr.Error("Charge un fichier audio.")

    try:
        target_path, residual_path, spans, _sampling_rate = _separate_to_files(
            audio_file, description, anchors_json, reranking_candidates, predict_spans
        )
    except json.JSONDecodeError as err:
        raise gr.Error(f"Anchors JSON invalide: {err}") from err
    except Exception as err:
        raise gr.Error(str(err)) from err

    return target_path, residual_path, "" if spans is None else str(spans)


async def verify_api_token(
    authorization: Annotated[str | None, Header()] = None,
    x_api_token: Annotated[str | None, Header(alias="X-API-Token")] = None,
):
    if not API_TOKEN:
        return

    bearer_prefix = "Bearer "
    bearer_token = None
    if authorization and authorization.startswith(bearer_prefix):
        bearer_token = authorization[len(bearer_prefix) :]

    if API_TOKEN not in {bearer_token, x_api_token}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _read_base64(path: str) -> str:
    with open(path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("ascii")


api = FastAPI(
    title="SAM-Audio API",
    version="1.0.0",
    description="Text-prompted audio source separation with SAM-Audio.",
)


@api.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@api.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(resolve_dtype()),
        "token_auth_enabled": bool(API_TOKEN),
    }


@api.post("/v1/audio/separations", dependencies=[Depends(verify_api_token)])
async def create_audio_separation(
    file: Annotated[UploadFile, File(description="Input audio file.")],
    description: Annotated[str, Form(description="Text prompt describing the sound to isolate.")],
    anchors: Annotated[
        str,
        Form(description='Optional JSON anchors, for example: [["+", 6.3, 7.0]]'),
    ] = "",
    reranking_candidates: Annotated[int, Form(ge=0, le=16)] = 0,
    predict_spans: Annotated[bool, Form()] = True,
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(prefix="sam-audio-upload-", suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        upload_path = tmp.name

    try:
        target_path, residual_path, spans, sampling_rate = _separate_to_files(
            upload_path, description, anchors, reranking_candidates, predict_spans
        )
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=422, detail=f"Invalid anchors JSON: {err}") from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    finally:
        try:
            os.unlink(upload_path)
        except FileNotFoundError:
            pass

    return {
        "model": MODEL_ID,
        "sampling_rate": sampling_rate,
        "target": {"content_type": "audio/wav", "base64": _read_base64(target_path)},
        "residual": {"content_type": "audio/wav", "base64": _read_base64(residual_path)},
        "spans": None if spans is None else str(spans),
    }


with gr.Blocks(title=f"SAM-Audio ({MODEL_ID})") as demo:
    gr.Markdown(
        """
# SAM-Audio — isolation de sons par prompt texte (et ancres temporelles)
- **Target** = le son demandé
- **Residual** = le reste
"""
    )

    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio (wav/mp3/ogg/…)")
        desc = gr.Textbox(
            label="Description (texte)",
            placeholder='Ex: "A man speaking" / "A dog barking" / "Piano playing a melody"',
            lines=2,
        )

    with gr.Row():
        anchors_json = gr.Textbox(
            label="Anchors (JSON) — optionnel",
            placeholder='Ex: [["+", 6.3, 7.0], ["-", 0.0, 2.0]]',
            lines=3,
        )

    with gr.Row():
        rerank = gr.Slider(
            0, 16, value=0, step=1,
            label="reranking_candidates (0 = off, + = mieux mais plus lent)"
        )
        predict_spans = gr.Checkbox(value=True, label="predict_spans")

    run = gr.Button("Séparer")

    with gr.Row():
        target_out = gr.Audio(label="Target (isolé)")
        residual_out = gr.Audio(label="Residual (reste)")

    spans_out = gr.Textbox(label="Spans (si dispo)", lines=4)

    run.click(
        separate,
        inputs=[audio_in, desc, anchors_json, rerank, predict_spans],
        outputs=[target_out, residual_out, spans_out],
    )


app = mount_gradio_app(api, demo, path="/ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", os.environ.get("SERVER_PORT", "7860"))))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
