# FP32 GPU inference and batching

The batched ONNX path uses **no PyTorch, FP16, TF32, or weight quantization**.
It reuses the released trained weights without retraining. The browser demo keeps
using the original single-request WASM model; these additional graphs are for
native CPU/CUDA deployments.

## Measured results

Test hardware: **NVIDIA GeForce RTX 5090, 32 GB**, driver 580.178.04, Linux,
Python 3.12.3, ONNX Runtime GPU 1.32.0 with CUDA 13 libraries. The environment
contained no PyTorch installation. **These are not RTX PRO 4000 measurements.**

Batch latency includes pronunciation, CPU/GPU transfers, synthesis and audio
returned to CPU memory. Models were loaded and warmed up first. Each row has
20 measured repeats, using two passages and both voices, repeated to fill the batch. Speaking speeds were calibrated
to approximately five or ten seconds per clip; actual lengths are in the report.

| Audio length per request | Batch size | Median batch latency | p95 batch latency |
|---|---:|---:|---:|
| ~5 seconds | 1 | 23.5 ms | 23.6 ms |
| ~5 seconds | 8 | 32.7 ms | 33.0 ms |
| ~5 seconds | 16 | 44.2 ms | 44.4 ms |
| ~5 seconds | 32 | 69.7 ms | 70.0 ms |
| ~10 seconds | 1 | 26.2 ms | 26.3 ms |
| ~10 seconds | 8 | 47.9 ms | 48.2 ms |
| ~10 seconds | 16 | 73.3 ms | 74.5 ms |
| ~10 seconds | 32 | 157.4 ms | 161.2 ms |

The 32-request, ten-second case was automatically split into two decoder groups
by the default padded-frame memory budget. With 16 ten-second requests, the new
batch path took about **73 ms**, compared with about **1,269 ms** to run those
requests serially through the original CUDA model: approximately **17.3×** faster
for that workload. The original comparison used the same precision and texts.

The microbatch queue was also tested with **64 requests submitted together**,
a maximum collection size of 32 and a 2 ms collection window:

| Clip length | All 64 requests completed | Completed within 0.5 seconds |
|---|---:|---:|
| ~5 seconds | 157 ms | 64 / 64 |
| ~10 seconds | 343 ms | 64 / 64 |

These burst results are a single controlled burst per clip-length setting,
not a production SLA or a maximum-capacity measurement. WAV encoding, HTTP,
network transfer, cold starts and unrelated GPU workloads are excluded. Larger
or more varied inputs and a longer queue change completion times. Throughput
and simultaneous-request deadlines must not be treated as interchangeable.

The largest sampled per-process GPU residency after a benchmark run was
**4,106 MiB**, including the comparison model's session. This is not a sampled
peak, a minimum VRAM requirement, or a guarantee for other batch/frame limits.
Test on your own RTX PRO 4000 before promising the same latency.

Raw data: [GPU benchmark](evaluation/gpu_batch_benchmark.json),
[GPU parity](evaluation/batched_validation_cuda.json),
[CPU parity](evaluation/batched_validation.json).

## Install and generate a batch

Use a fresh Python environment with an NVIDIA driver compatible with CUDA 13.
The GPU requirements install CUDA/cuDNN runtime libraries through pip; they do
not install a training framework. Do not install both `onnxruntime` and
`onnxruntime-gpu` in the same environment. The repository's training-package
installation is unnecessary for this standalone inference path.

```bash
git clone https://github.com/ub1979/little-tts
cd little-tts
python -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-onnx-gpu.txt

python batched_onnx.py --device cuda --batch-size 16 \
  --text "The morning sunlight streamed through the window, warming the quiet room." \
  --text "The garden was quiet, and the air was filled with the scent of roses." \
  --output batch-audio
```

To deploy only the GPU inference files from Hugging Face, rather than clone
the training repository, use the Hub CLI:

```bash
pip install 'huggingface-hub>=1.16,<2'
hf download TuringLogic/little-tts --local-dir little-tts-gpu \
  --include 'onnx/batched/*' 'phonemizer/*' 'batched_onnx.py' 'batch_queue.py' \
  'requirements-onnx-gpu.txt' 'LICENSE*' 'NOTICE'
```

The original CPU/browser graph is still `onnx/model.onnx`. The new graphs are
`onnx/batched/encoder.onnx` and `onnx/batched/decoder.onnx`, with metadata in
`onnx/batched/config.json`. All learned floating-point weights and audio outputs remain FP32.

```python
from batched_onnx import BatchedSynthesizer

synth = BatchedSynthesizer(".", device="cuda", max_batch=16,
                           max_padded_frames=16000)
results = synth.speak_batch(
    ["The garden was quiet, and the air was filled with roses.",
     "The road curved gently through the hills."],
    voices=["92", "6097"],
    speeds=[1.0, 1.0],
    seeds=[0, 1],
)
for waveform, durations in results:
    print(len(waveform) / synth.sample_rate)
print(synth.last_stats)
```

Results are restored to input order, even when requests are length-sorted.
Voice 92 remains the recommended default. `--device cuda` fails if CUDA cannot
initialize; it does not silently switch the full session to CPU. Small shape
operations can still be assigned to CPU by ONNX Runtime.

## Combine independent callers

`speak_batch()` processes a supplied list. For independent concurrent callers,
use the bounded queue to collect requests into actual shared batches:

```python
from batch_queue import BatchQueue

with BatchQueue(synth, max_wait_ms=2, max_pending=256) as queue:
    first = queue.submit("The morning was quiet, and the sky was clear.")
    second = queue.submit("The garden was filled with the scent of roses.")
    first_wave, _ = first.result()
    second_wave, _ = second.result()
```

Use one queue per synthesizer. Direct concurrent calls to `speak_batch()` are
serialized to protect shared runtime state; the queue is what combines them.
Requests receive results at batch completion. Queue capacity is bounded, and
overload raises an error rather than accumulating unlimited work. Per-call
settings are validated; inference failures are propagated to the affected batch.

`max_batch` controls collection size. `max_padded_frames` limits
`batch_size × longest_predicted_frame_count` before allocating DSP tensors; an
oversized group is split automatically. This is a memory-control heuristic,
not an exact VRAM allocation limit. Each request is limited to 512 phoneme
tokens and 3,000 frames; split longer input into passages. For low latency with
ten-second clips, start with `max_batch=16` and measure the actual workload.

## What changed

- Real dynamic batches, token masks and frame masks replace batch-size-one
  assumptions. Each request keeps its own voice, speed, pitch controls, noise
  seed, durations and output length; padding cannot leak into real audio.
- Text encoding and waveform decoding are separate graphs. Only small duration
  arrays return to the host to select the frame extent. I/O binding keeps the
  large encoder hidden tensor on CUDA between stages.
- RoPE positions and other shape inputs come from cached host-defined ranges,
  avoiding duration-dependent GPU-to-host shape discovery inside a monolithic
  graph. Runtime graph optimization is enabled.
- The harmonic oscillator is evaluated in 16-harmonic tiles to bound temporary
  memory. The noise filter uses the original real-valued DFT matrices as dense
  matrix multiplications and four-way overlap-add, avoiding the slow DSP
  convolution/transposed-convolution route measured on the test GPU.
- CUDA uses the NC1D convolution layout, and TF32 is explicitly disabled.
  No TensorRT, PyTorch or quantization is required.

## Rebuild and validate without PyTorch

The checked-in graphs are ready to use. To rebuild them, install the small
additional graph-building requirement:

```bash
pip install -r requirements-onnx-build.txt
python scripts/build_batched_onnx.py
python scripts/validate_batched_onnx.py --device cuda \
  --output evaluation/my_gpu_validation.json
```

The builder extracts unchanged learned tensors directly from `onnx/model.onnx`,
not from a PyTorch checkpoint. It records that source file's SHA256. The parity
checker exposes the original ONNX graph's noise input and supplies identical
noise to compare waveform samples. CPU and GPU checks cover both voices,
mixed lengths/speeds/pitch settings, input-order restoration, batch invariance
and forced memory-budget splitting. All tested token durations and output
lengths matched. Maximum GPU waveform error was approximately **0.1% of reference
peak amplitude**; this numerical check is not a human naturalness rating.

To reproduce the latency and simultaneous-burst tests:

```bash
python scripts/benchmark_batched_onnx.py --device cuda \
  --batch-sizes 1 2 4 8 16 32 --seconds 5 10 \
  --repeats 20 --serial-repeats 3 --burst 64 \
  --output evaluation/my_gpu_benchmark.json
```

For CPU-only validation, use `requirements-onnx.txt` and
`requirements-onnx-build.txt` in a separate environment, then pass `--device cpu`.
The existing browser demo does not use or download the batched graphs.

Weights, including these derived graphs, are CC BY-NC 4.0. Source code is
Apache-2.0; third-party pronunciation assets retain their included notices.
