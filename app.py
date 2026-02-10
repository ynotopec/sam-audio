import argparse
import os
import json
import tempfile
import torch
import torchaudio
import gradio as gr

from sam_audio import SAMAudio, SAMAudioProcessor

from huggingface_hub import login

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    login(token=_HF_TOKEN)

MODEL_ID = os.environ.get("SAM_AUDIO_MODEL", "facebook/sam-audio-large-tv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None


#def get_model():
#    global _model, _processor
#    if _model is None or _processor is None:
#        _model = SAMAudio.from_pretrained(MODEL_ID).to(DEVICE).eval()
#        _processor = SAMAudioProcessor.from_pretrained(MODEL_ID)

#        _model = SAMAudio.from_pretrained(
#            MODEL_ID,
#            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
#            proxies=None,
#            resume_download=False,
#        ).to(DEVICE).eval()

#        _processor = SAMAudioProcessor.from_pretrained(
#            MODEL_ID,
#            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
#            proxies=None,
#            resume_download=False,
#        )

#    return _model, _processor

def get_model():
    global _model, _processor
    if _model is None or _processor is None:
        _model = SAMAudio.from_pretrained(
            MODEL_ID,
        ).to(DEVICE).eval()

        _processor = SAMAudioProcessor.from_pretrained(
            MODEL_ID,
        )
    return _model, _processor


def parse_anchors(anchors_json: str):
    """
    Attendu: JSON de la forme:
    [
      ["+", 6.3, 7.0],
      ["-", 0.0, 2.0]
    ]
    ou vide => None
    """
    anchors_json = (anchors_json or "").strip()
    if not anchors_json:
        return None
    anchors = json.loads(anchors_json)
    # Le processor attend anchors=[anchors] pour un seul audio (liste de listes)
    return [anchors]


@torch.inference_mode()
def separate(audio_file, description, anchors_json, reranking_candidates, predict_spans):
    if audio_file is None:
        raise gr.Error("Charge un fichier audio.")
    description = (description or "").strip()
    if not description:
        raise gr.Error("Donne une description texte (ex: 'A man speaking', 'A dog barking').")

    model, processor = get_model()

    anchors = None
    try:
        anchors = parse_anchors(anchors_json)
    except Exception as e:
        raise gr.Error(f"Anchors JSON invalide: {e}")

    # Gradio fournit un chemin local
    audio_path = audio_file

    inputs = processor(
        audios=[audio_path],
        descriptions=[description],
        anchors=anchors,
    ).to(DEVICE)

    kwargs = {}
    if reranking_candidates and int(reranking_candidates) > 0:
        kwargs["reranking_candidates"] = int(reranking_candidates)

    result = model.separate(inputs, predict_spans=bool(predict_spans), **kwargs)

    # Sauver en wav (target + residual)
    sr = processor.audio_sampling_rate
    tmpdir = tempfile.mkdtemp(prefix="sam-audio-")

    target_path = os.path.join(tmpdir, "target.wav")
    residual_path = os.path.join(tmpdir, "residual.wav")

    torchaudio.save(target_path, result.target[0].unsqueeze(0).cpu(), sr)
    torchaudio.save(residual_path, result.residual[0].unsqueeze(0).cpu(), sr)

    spans_txt = ""
    if hasattr(result, "spans") and result.spans is not None:
        spans_txt = str(result.spans)

    return target_path, residual_path, spans_txt


with gr.Blocks(title="SAM-Audio (facebook/sam-audio-large-tv)") as demo:
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

#demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port)
