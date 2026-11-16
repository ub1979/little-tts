"""Measure FP32 batch latency, serial CUDA comparison, and a simultaneous burst.

Timings include phonemization, host/device transfers and audio returning to CPU.
They exclude loading/warm-up, WAV encoding, HTTP/network time and queueing unless
explicitly marked as the burst test.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batched_onnx import BatchedSynthesizer
from batch_queue import BatchQueue
from onnx_infer import Synthesizer

SHORT = ["The morning sunlight streamed through the window, warming the quiet room.",
         "The road curved gently through the hills, leading us toward the distant village."]
LONG = ["After a long and difficult journey across the mountains, the travellers finally reached the small village, where they were welcomed with warm food, a place to sleep and many questions about the road behind them.",
        "She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word. The morning sunlight streamed through the window, warming the quiet room."]


def gpu_info():
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
        return text
    except (OSError, subprocess.SubprocessError): return None


def gpu_resident_mib():
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
        return sum(float(line.split(",")[1]) for line in text.splitlines() if int(line.split(",")[0]) == os.getpid())
    except (OSError, ValueError, subprocess.SubprocessError): return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=ROOT)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--seconds", type=float, nargs="+", default=[5, 10])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--serial-repeats", type=int, default=3)
    p.add_argument("--burst", type=int, default=32)
    p.add_argument("--output", type=Path, default=ROOT / "evaluation/gpu_batch_benchmark.json")
    args = p.parse_args()
    synth = BatchedSynthesizer(args.model_dir, args.device, max_batch=max(args.batch_sizes))
    original = Synthesizer(args.model_dir)
    if args.device == "cuda":
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        original.session = ort.InferenceSession(str(args.model_dir/"onnx/model.onnx"), options,
            providers=[("CUDAExecutionProvider", {"use_tf32": 0, "cudnn_conv_algo_search": "DEFAULT",
                                                   "cudnn_conv_use_max_workspace": "0"}), "CPUExecutionProvider"])
        if original.session.get_providers()[0] != "CUDAExecutionProvider": raise RuntimeError("Original CUDA session fell back to CPU")
    results, bursts = [], []
    for target in args.seconds:
        passages = SHORT if target <= 6 else LONG
        catalog = []
        for text in passages:
            for voice in synth.voices:
                speed = 1.0
                for _ in range(3):
                    wave, _ = synth.speak_batch([text], voices=[voice], speeds=[speed])[0]
                    speed *= len(wave)/synth.sample_rate/target
                    speed = min(2., max(.5, speed))
                catalog.append((text, voice, speed))
        for size in args.batch_sizes:
            requests = [catalog[j % len(catalog)] for j in range(size)]
            texts, voices, speeds = map(list, zip(*requests))
            for _ in range(3): synth.speak_batch(texts, voices=voices, speeds=speeds)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = synth.speak_batch(texts, voices=voices, speeds=speeds)
                times.append(time.perf_counter()-start)
            serial_times = []
            for text, voice, speed in requests[:1]: original.speak(text, voice, speed)
            for _ in range(args.serial_repeats):
                start = time.perf_counter()
                for text, voice, speed in requests: original.speak(text, voice, speed)
                serial_times.append(time.perf_counter()-start)
            audio = sum(len(wave) for wave, _ in output)/synth.sample_rate
            median = statistics.median(times)
            row = dict(target_clip_seconds=target, batch_size=size, actual_mean_clip_seconds=audio/size,
                       median_batch_ms=median*1000, p95_batch_ms=float(np.percentile(times, 95))*1000,
                       min_batch_ms=min(times)*1000, max_batch_ms=max(times)*1000,
                       clips_per_second=size/median, audio_seconds_per_second=audio/median,
                       original_serial_median_ms=statistics.median(serial_times)*1000,
                       speedup_over_original_serial=statistics.median(serial_times)/median,
                       gpu_resident_mib_after_run=gpu_resident_mib(), repeats=len(times),
                       timings_ms=[x*1000 for x in times])
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!="timings_ms"}), flush=True)
            report = dict(measured_at=datetime.now(timezone.utc).isoformat(), device=args.device,
                          gpu=gpu_info(), platform=platform.platform(), python=platform.python_version(),
                          onnxruntime=ort.__version__, precision="FP32", tf32=False, quantized=False,
                          pytorch_installed=importlib.util.find_spec("torch") is not None,
                          pytorch_imported="torch" in sys.modules, results=results, simultaneous_bursts=bursts,
                          scope=__doc__, memory_note="Per-process resident VRAM sampled after each run; not a sampled peak. Includes both comparison sessions.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
        if args.burst:
            completed = []
            with BatchQueue(synth, max_wait_ms=2) as queue:
                started = time.perf_counter()
                futures = []
                for j in range(args.burst):
                    text, voice, speed = catalog[j % len(catalog)]
                    f = queue.submit(text, voice, speed)
                    f.add_done_callback(lambda _: completed.append(time.perf_counter()-started))
                    futures.append(f)
                for f in futures: f.result()
            burst = dict(target_clip_seconds=target, simultaneous_requests=args.burst,
                         max_batch=synth.max_batch, queue_wait_ms=2,
                         p50_completion_ms=float(np.percentile(completed, 50))*1000,
                         p95_completion_ms=float(np.percentile(completed, 95))*1000,
                         all_completed_ms=max(completed)*1000,
                         finished_within_90ms=sum(t<=.09 for t in completed),
                         finished_within_500ms=sum(t<=.5 for t in completed))
            bursts.append(burst)
            print("BURST",json.dumps(burst),flush=True)
            report["simultaneous_bursts"] = bursts
            args.output.write_text(json.dumps(report, indent=2)+"\n")
    assert "torch" not in sys.modules


if __name__ == "__main__": main()
