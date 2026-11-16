"""Measure real-time factor and peak process RAM for this exact CPU release."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
os.environ["LITTLE_TTS_FRONTEND"] = "own"
os.environ["DIZIONE_DATA"] = str(ROOT / "phonemizer")
import torch
from little_tts.infer import load_model, speak

TEXTS = ["Hello! This is a sample of my voice as I learn to speak naturally.",
         "The morning sunlight streamed through the window, warming the quiet room.",
         "At half past seven, the station was already busy. Travelers checked their tickets while a gentle rain fell outside."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/cpu_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    load_start = time.perf_counter()
    model = load_model(ROOT / "inference.pt", "cpu")
    load_seconds = time.perf_counter() - load_start
    records = []
    with torch.inference_mode():
        first = time.perf_counter()
        wave, _, _ = speak(model, TEXTS[0], model.config.speakers[0], seed=0)
        first_seconds = time.perf_counter() - first
        startup_seconds = time.perf_counter() - START
        for voice in model.config.speakers:
            # Warm both voices before the timed measurements.
            speak(model, TEXTS[0], voice, seed=0)
            for text in TEXTS:
                for repeat in range(args.repeats):
                    began = time.perf_counter()
                    wave, _, _ = speak(model, text, voice, seed=0)
                    seconds = time.perf_counter() - began
                    duration = len(wave) / model.config.sample_rate
                    records.append(dict(voice=voice, text=text, repeat=repeat,
                                        synthesis_seconds=seconds, audio_seconds=duration,
                                        real_time_factor=seconds/duration))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if platform.system() == "Darwin" else platform.processor()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    total_time = sum(r["synthesis_seconds"] for r in records)
    total_audio = sum(r["audio_seconds"] for r in records)
    report = dict(measured_at=datetime.now(timezone.utc).isoformat(), cpu=cpu,
                  platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                  threads=args.threads, interop_threads=1, device="cpu", precision="float32",
                  model_load_seconds=load_seconds, first_synthesis_seconds=first_seconds,
                  import_load_and_first_synthesis_seconds=startup_seconds,
                  peak_process_rss_mib=peak_bytes/1024**2,
                  utterances=len(records), total_synthesis_seconds=total_time, total_audio_seconds=total_audio,
                  weighted_real_time_factor=total_time/total_audio,
                  times_faster_than_real_time=total_audio/total_time,
                  median_utterance_rtf=statistics.median(r["real_time_factor"] for r in records),
                  min_utterance_rtf=min(r["real_time_factor"] for r in records),
                  max_utterance_rtf=max(r["real_time_factor"] for r in records), records=records,
                  scope="Includes phonemization and waveform synthesis; excludes file encoding, Gradio, queue/network time. Peak RSS includes Python, PyTorch, checkpoint loading, pronunciation assets, and synthesis.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
