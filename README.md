---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-to-speech
library_name: onnxruntime
tags:
- text-to-speech
- ddsp
- onnx
- cpu
datasets:
- MikhailT/hifi-tts
model-index:
- name: Little TTS
  results: []
---

# Little TTS

A **4,648,907-parameter English TTS model** with a DDSP harmonic-plus-noise
waveform generator, two voices (`92`, `6097`), and **24 kHz mono output**.
Trained from scratch on approximately 29.6 hours of real-human HiFi-TTS speech.
**CPU inference needs no PyTorch and no GPU.** The release uses FP32 ONNX;
no weight quantization is applied.

- **Try it:** https://huggingface.co/spaces/TuringLogic/little-tts
- **Model:** https://huggingface.co/TuringLogic/little-tts
- **Training and inference code:** https://github.com/ub1979/little-tts
- **Weights: CC BY-NC 4.0 — noncommercial use only.** See `LICENSE-WEIGHTS`.
- Source code: Apache-2.0. Dataset and pronunciation assets have separate terms;
  see `NOTICE`, `DATA_LICENSE.txt`, and `phonemizer/LICENSE.*`.

## Size, speed and RAM

| Item | Measured size |
|---|---:|
| Inference parameters, including waveform head | 4,648,907 |
| FP32 PyTorch checkpoint, for training/export workflows | 18.66 MB |
| FP32 ONNX model, for inference without PyTorch | 19.52 MB |
| Pronunciation files | About 5.4 MB |
| ONNX model + pronunciation files | About 25 MB, excluding runtime packages |

The ONNX file is slightly larger because it includes the exported inference graph
and DSP constants. It removes the **PyTorch runtime dependency**, rather than
compressing or quantizing the neural weights. Download only the ONNX files and
pronunciation assets if you do not need the training checkpoint or source.

**Local CPU benchmark, 2026-11-16:** Apple M4 Pro, macOS 15.1, 24 GiB system RAM,
two inference threads, one inter-op thread, Python 3.12.2. Each runtime was tested
in a separate process on the same three sentences, both voices, three timed
repeats after warm-up: 18 utterances and 97.472 seconds of generated audio.

| Runtime | Real-time factor ↓ | Faster than real time ↑ | Peak process RAM |
|---|---:|---:|---:|
| **ONNX Runtime 1.32.0, FP32** | **0.0091** | **109.9×** | **501.6 MiB** |
| PyTorch 2.15.0, FP32 | 0.0109 | 91.9× | 680.1 MiB |

**RTF = generation seconds / audio seconds.** RTF 0.0091 means approximately
0.091 seconds to synthesize 10 seconds of audio on this test machine after warm-up.
This is throughput, not streaming latency. The model generates a complete utterance.
ONNX model load took 0.110 seconds; first synthesis took 0.115 seconds. Script
imports, model load and first synthesis together took 1.045 seconds from script
execution (not including Python interpreter startup).

Peak RSS includes Python, the inference runtime, pronunciation assets, model
loading and waveform generation, not just neural weights. It is a measured peak
for these inputs, **not a minimum RAM requirement or a bound for longer inputs**.
For deployment planning, allow at least 1 GiB per worker and more for a web UI,
longer inputs or concurrent requests; this is headroom guidance, not a tested
minimum. CPU architecture, runtime versions and load affect speed and RAM.
The public Space runs ONNX Runtime Web on each visitor’s device; these native
Mac measurements are not browser benchmarks. Gradio, WAV encoding, queue time and network time are excluded.

Raw measurements and reproducible scripts:
`evaluation/onnx_cpu_benchmark.json`, `evaluation/cpu_benchmark.json`,
`scripts/benchmark_onnx.py`, `scripts/benchmark_cpu.py`.

## Inference without PyTorch

```bash
git clone https://github.com/ub1979/little-tts
cd little-tts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-onnx.txt
python onnx_infer.py --voice 92 \
  --text "The morning sunlight streamed through the window." --output speech.wav
```

This path installs NumPy, ONNX Runtime, SoundFile and the pinned NumPy-based
`dizione` phonemizer. **It does not install or import PyTorch.** Use `--voice 6097`
for the other voice, `--speed 0.9` for slower speech, or `--threads 2` to set CPU
threads. Keep requests to short English sentences; the graph accepts at most
512 phoneme tokens and 3,000 acoustic frames (approximately 32 seconds).

To download only the inference assets from Hugging Face:

```bash
pip install 'huggingface-hub>=1.16,<2'
hf download TuringLogic/little-tts --local-dir model \
  --include 'onnx/*' 'phonemizer/*' 'onnx_infer.py' 'requirements-onnx.txt' \
  'LICENSE*' 'NOTICE' 'DATA_LICENSE.txt'
pip install -r model/requirements-onnx.txt
python model/onnx_infer.py --text "Hello, this runs on a CPU." --output hello.wav
```

## Training code and recipe

The complete model, losses, data preparation, forced alignment, trainer,
evaluation and export code are in `src/little_tts/`. The real-human training
pipeline is `scripts/train_human_5090.py`. See [TRAINING.md](TRAINING.md) for
installation, dataset download, training, resume, export and evaluation.
**PyTorch is needed for training/export only.** The public demo uses ONNX Runtime Web in the browser; its source is in `web/`.
The optional Gradio starter in `demo/` also uses ONNX and requires server hosting.

Final checkpoint: **104,000 steps**, 37,833 accepted training clips after
preparation/alignment filtering; approximately 29.617 hours. Data preparation
selected 38,656 clips before filtering, as recorded in `data_provenance.json`.
The exact run settings and dataset fingerprints are in `recipe.json`.
Training loop: 12,392.9 seconds (about 3 h 27 min) on the training server,
excluding download, preparation, alignment and evaluation. This is not a CPU
training-time estimate. Optimizer/resume state and training audio are not included.

## Recommended voice and examples

Voice **92** is selected by default and recommended for this demo. Voice 6097
remains available as an alternate. The default is the sea story, followed by
Morning, Garden, Journey and Evening examples. These use plain narrative text.

The original examples were checked with the pronunciation engine and local
Whisper large-v3-turbo ASR. Numbers, dates, currency, and read/live pronunciations
were correct; the transcripts recovered the intended words and the waveforms
did not clip. The complaint is therefore not explained by a confirmed general
text-normalization failure. Naturalness and speaker quality remain limitations.

The browser now keeps short sentences together in one generation, preserving
context within a 256-phoneme budget. Longer requests split at phrase boundaries.
The technical greeting, numbers, question and homograph challenge buttons were
replaced by the narrative examples. This is an inference/demo update; it does
not change the trained weights or establish a human-rated quality improvement.
See `evaluation/example_audit.json` for the checks.

## Speech quality

The waveform head uses 64 predicted sinusoidal harmonics plus filtered noise.

Existing intelligibility checks used faster-whisper small.en (CPU/int8 ASR).
These are **mean sentence WER**, not corpus WER or human naturalness scores:

| Group | Clips | Generated WER | Real-reference WER |
|---|---:|---:|---:|
| Validation | 32 | 13.45% | 4.85% |
| Test | 32 | 7.33% | 7.10% |
| Novel sentences | 16 | 1.67% | — |

These scores were measured on the PyTorch checkpoint. They compare generated
speech with real recordings. Names and some short
phrases can be mispronounced or misrecognized. The sample size is small.
English only; no Urdu voice model is included.

## ONNX validation

The FP32 export was checked against PyTorch using identical externally supplied
noise on five varied sentences in both voices. All ten checks matched durations
and output length. Maximum waveform error was below 0.003 of reference peak
amplitude. The normal ONNX export also produced finite audio for all ten cases.
See `evaluation/onnx_validation.json`. The released graph generates its own noise,
so repeated synthesis is not sample-identical. No perceptual quality claim follows
from numerical parity alone.

## Attribution and licences

Inspired by [TinyVoca](https://github.com/n4ze3m/tiny-voca) by Muhammed Nazeem (Apache-2.0).
The weights were trained from scratch.
Source changes include the DDSP head, training/export adjustments, real-human
recipe, and demo. `LICENSE` / `LICENSE-CODE` cover source; `LICENSE-WEIGHTS` covers
`inference.pt` and `onnx/model.onnx`. Licence metadata refers to the new weights.

Training data: [HiFi-TTS](https://www.openslr.org/109/), Bakhturina, Lavrukhin,
Ginsburg and Zhang (2021), CC BY 4.0; mirrored at `MikhailT/hifi-tts`, pinned revision
`f0a934ddf4e8da0fc6c53fb42fda2c9ac8ed93df`. See `DATA_LICENSE.txt` for selection,
resampling and alignment modifications. Source data retains its own licence.
The pronunciation assets retain CMUdict and homograph-data notices.

Checkpoint SHA256:
`879c8f29d9228439818e04b04fd3522abaecb262caf199d3f9cc1f1a7afa1334`.
`SHA256SUMS` covers the release files. No private Urdu recordings, private
transcripts, credentials or raw training recordings are part of this release.
