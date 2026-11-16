# Train and export Little TTS

Source code is Apache-2.0. The supplied trained weights and ONNX export are
CC BY-NC 4.0. Source data and pronunciation assets retain their original licences.
This document describes the released English recipe, not an Urdu training run.

## 1. Environment

Use Linux, Python 3.11 or 3.12, sufficient disk space for the clean HiFi-TTS
shards, and an NVIDIA GPU for the original preparation/alignment/training workflow.
Install a compatible PyTorch + torchaudio pair for your CUDA version first;
then install this repository and the extras:

```bash
python -m venv .venv
source .venv/bin/activate
# Install matching torch / torchaudio for your GPU from pytorch.org first.
pip install -e '.[own,prepare,annotate,evaluate]'
export LITTLE_TTS_FRONTEND=own
export DIZIONE_DATA="$PWD/phonemizer"
```

The own extra pins the exact dizione revision used for the released model.
Do not train against a different phoneme inventory/front end and expect existing
annotations or weights to remain compatible. Optional Urdu hooks in source need
the separate urdu-phonemizer project; it is not required for this English recipe.

## 2. Get the public source data

Read `DATA_LICENSE.txt` before downloading. The training pipeline expects the
37 clean parquet shards at this pinned revision:

```bash
hf download MikhailT/hifi-tts --repo-type dataset \
  --revision f0a934ddf4e8da0fc6c53fb42fda2c9ac8ed93df \
  --include 'data/*.clean-*.parquet' --local-dir human_data/hifitts
```

No recordings or annotations are bundled in this release. The recipe selects
speakers 92 and 6097, caps each at about 15 hours, excludes duplicated held-out
texts, resamples to 24 kHz and preserves the original validation/test split.
Preparation and alignment quality filters reduce the selected training count
from 38,656 to the released run's 37,833 accepted clips. Counts can vary with
software/hardware and numerical filtering behavior; consult saved journals.

## 3. Prepare, align, and train

```bash
PREPARE_WORKERS=4 python scripts/train_human_5090.py
```

This script creates `human_data/hifitts_run/`, builds acoustic features, downloads
the wav2vec2 phoneme recognizer for offline alignment, and trains a new DDSP model
from scratch to 104,000 steps. It validates the aligner's current main revision
against pinned revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4` and stops if it
has changed. Review and pin an alternative explicitly if that happens.
The recognizer is used only for labels; it supplies no TTS generator weights.

The script starts or resumes training in `runs/ddsp_hifitts_human/`, writes
`last.pt` for resume and `inference.pt` for synthesis, and saves every 1,000 steps.
It automatically resumes if `last.pt` exists. Start in a new workspace to train
a fresh run. To request a clean stop, create `STOP_HUMAN_TRAINING` at the repository
root; deliberately remove it before restarting.

The script is the original run driver with only release asset paths adjusted
from `weights/phonemizer` and `docs/HUMAN_DATA_LICENSE.txt` to the release layout.
`config.json` records architecture; `recipe.json` records actual optimization
settings, loss weights, dataset fingerprints and final accepted utterance count.
The run was trained with a 9,000-frame batch budget and max batch size 32, learning
rate 0.0003, warm-up 1,000, GAN start 8,000, and a duration/prosody teacher schedule
that reaches fully predicted targets. The trainer provides other controls:

```bash
python -m little_tts.train --help
python -m little_tts.annotate --help
```

Reducing frame/batch budgets can lower GPU memory demand. The training server
used an RTX 5090; its runtime does not predict performance on another GPU or CPU.
No full training run was rerun as part of release packaging.

## 4. Evaluate and export

```bash
python -m little_tts.evaluate \
  --checkpoint runs/ddsp_hifitts_human/inference.pt \
  --annotations human_data/hifitts_run/annotations \
  --output eval/new_run --tag new-run --splits validation test \
  --per-voice 16 --novel 8 --device cpu

pip install 'onnx>=1.17,<2' 'onnxruntime==1.32.0'
python -m little_tts.export \
  --checkpoint runs/ddsp_hifitts_human/inference.pt --output onnx/new_model.onnx
```

The exporter also saves matching metadata JSON. Keep the ONNX graph, matching
metadata and training pronunciation data together. For a numerically controlled
export check, create a second graph with `--external-noise` and compare it:

```bash
python -m little_tts.export --checkpoint runs/ddsp_hifitts_human/inference.pt \
  --output /tmp/exact.onnx --external-noise
python scripts/check_onnx.py --checkpoint runs/ddsp_hifitts_human/inference.pt \
  --onnx /tmp/exact.onnx --release onnx/new_model.onnx --threads 2
```

Use the standalone `onnx_infer.py` and `requirements-onnx.txt` for deployment.
**PyTorch is not needed for ONNX inference or the public demo.**

For the supplied release's benchmarks:

```bash
python scripts/benchmark_onnx.py
PYTHONPATH=src python scripts/benchmark_cpu.py
```

The second command requires PyTorch. ASR WER checks intelligibility; use blinded,
loudness-matched listening tests for naturalness and preference.
