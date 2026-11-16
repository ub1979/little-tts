"""Render consistent CPU listening previews from an inference checkpoint."""
import argparse
import json
from pathlib import Path
import soundfile as sf
import torch
from little_tts.infer import load_model, speak
from little_tts.text import set_frontend

TEXTS = (
    "Hello! This is a sample of my voice as I learn to speak naturally.",
    "The morning sunlight streamed through the window, warming the quiet room.",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    set_frontend("own")
    model = load_model(args.checkpoint, "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for voice in model.config.speakers:
            for index, text in enumerate(TEXTS, 1):
                wave, phones, _ = speak(model, text, voice, seed=0)
                peak = float(wave.abs().max())
                # Attenuate peaks above .95 for playback, without boosting quiet output.
                gain = min(1., .95 / max(peak, 1e-9))
                path = args.output / f"voice_{voice}_{index}.wav"
                sf.write(path, wave.numpy() * gain, model.config.sample_rate, subtype="PCM_16")
                records.append(dict(voice=voice, text=text, file=path.name,
                    seconds=len(wave)/model.config.sample_rate, raw_peak=peak, playback_gain=gain))
    (args.output / "samples.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Listening samples ready:", args.output, flush=True)


if __name__ == "__main__":
    main()
