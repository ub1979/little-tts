"""Public CPU demo for the released English DDSP checkpoint."""
import os
from pathlib import Path
import sys
import time

import gradio as gr
from huggingface_hub import snapshot_download
import numpy as np

MODEL_ID = "TuringLogic/little-tts"
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MODEL_DIR = Path(os.environ["MODEL_DIR"]) if "MODEL_DIR" in os.environ else Path(
    snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                      allow_patterns=["onnx/model.onnx", "onnx/model.json", "onnx_infer.py", "phonemizer/*", "samples/*"]))
os.environ["DIZIONE_DATA"] = str(MODEL_DIR / "phonemizer")
sys.path.insert(0, str(MODEL_DIR))
from onnx_infer import Synthesizer

model = Synthesizer(MODEL_DIR, threads=2)


def generate(text, voice, speed):
    text = text.strip()
    if not text:
        raise gr.Error("Enter a sentence to generate speech.")
    if len(text) > 350:
        raise gr.Error("Please use up to 350 characters per request.")
    if voice not in model.voices:
        raise gr.Error("Choose one of the two available voices.")
    if not 0.8 <= float(speed) <= 1.2:
        raise gr.Error("Choose a speed between 0.8 and 1.2.")
    started = time.perf_counter()
    try:
        audio, _ = model.speak(text, voice, speed=float(speed))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    peak = float(np.abs(audio).max())
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    # PCM avoids Gradio's float-array peak normalization, preserving model volume.
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype(np.int16)
    duration = len(pcm) / model.sample_rate
    elapsed = time.perf_counter() - started
    return (model.sample_rate, pcm), f"{duration:.1f}s audio · {elapsed:.2f}s generation · RTF {elapsed/duration:.3f} · ONNX Runtime on CPU"


TEXTS = ['She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word.', 'The morning sunlight streamed through the window, warming the quiet room.', 'The garden was quiet, and the air was filled with the scent of roses.', 'The road curved gently through the hills, leading us toward the distant village.', 'As evening fell, the last light of the day faded slowly from the sky.']


with gr.Blocks(title="Little TTS · English TTS") as demo:
    gr.Markdown("# Little TTS\nSmall English text-to-speech · **4.65M parameters** · two voices · 24 kHz audio")
    gr.Markdown("**Noncommercial use only — model weights: CC BY-NC 4.0.** "
                "[Model & licence](https://huggingface.co/TuringLogic/little-tts) · "
                "[Source code](https://github.com/ub1979/little-tts)")
    with gr.Tab("Try the voices"):
        text = gr.Textbox(label="English text", lines=3, max_length=350,
                          value=TEXTS[0])
        with gr.Row():
            voice = gr.Dropdown(choices=[("Voice 92 · recommended", "92"), ("Voice 6097 · alternate", "6097")], value="92", label="Voice")
            speed = gr.Slider(0.8, 1.2, value=1, step=0.05, label="Speaking speed")
        button = gr.Button("Generate speech", variant="primary")
        audio = gr.Audio(label="Generated speech", type="numpy", format="wav")
        status = gr.Textbox(label="Generation", interactive=False)
        gr.Examples([[example, "92", 1.0] for example in TEXTS], inputs=[text, voice, speed])
        button.click(generate, [text, voice, speed], [audio, status], concurrency_limit=1, api_name="generate")
        gr.Markdown("English only. Short sentences work best. Pronunciation and naturalness can vary, especially for names. "
                    "Requests run on shared CPU hardware and may queue. The app does not save a custom transcript log.")

if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=1).launch()
