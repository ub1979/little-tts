---
title: Little TTS
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
- TuringLogic/little-tts
short_description: Small English TTS that runs locally in your browser
---

# Little TTS demo

Browser CPU demo of [TuringLogic/little-tts](https://huggingface.co/TuringLogic/little-tts).
Runs on **ONNX Runtime Web, without PyTorch**, using unquantized FP32 weights.
The app source is Apache-2.0. **Model weights are CC BY-NC 4.0 (noncommercial use only)**;
see the model repository's `LICENSE-WEIGHTS`. Third-party pronunciation data retains
its included licences. No training recordings are included.

Generate English speech with voices `92` and `6097`, with adjustable speed and local audio playback.

## Size and CPU measurements

The model has **4,648,907 parameters**. The FP32 ONNX file is about **19.5 MB**;
pronunciation data adds about 5.4 MB. The optional PyTorch checkpoint is 18.66 MB
and is not downloaded or loaded by this app. No GPU is required.

Local benchmark: Apple M4 Pro, macOS 15.1, two inference threads, Python 3.12.2,
ONNX Runtime 1.32.0. Across 18 utterances after warm-up, **RTF 0.0091 (109.9×
real time)** and **501.6 MiB peak process RAM**. The equivalent PyTorch benchmark
used RTF 0.0109 (91.9×) and 680.1 MiB peak process RAM. RAM includes the runtime,
Python, pronunciation data and waveform synthesis. The ONNX peak is a measurement
for the test inputs, not a universal minimum or maximum. Allow at least 1 GiB per
worker as deployment headroom, plus UI/runtime overhead for native deployment; browser memory differs.

These are native Python measurements on the local Mac, not browser benchmarks. This free static Space runs entirely on the visitor’s device; it has no inference server.
Encoding, Gradio, queuing and network time are excluded. RTF means generation
seconds divided by audio seconds; lower is faster. See the model README and
`evaluation/*benchmark.json` for reproducible results and limitations.

The complete training recipe and source are available on
[GitHub](https://github.com/ub1979/little-tts). PyTorch is required for training
and export only. Neither quantization nor PyTorch is used for this demo.


The static demo uses a web worker, ONNX Runtime Web 1.20.1, and Pyodide 0.26.4 with NumPy for the pinned dizione phonemizer. First load downloads the model and runtimes; later loads can use the browser cache. Entered text is processed locally. No analytics are added by this application.

## Run the static demo locally

From the repository root:

```bash
python web/build_assets.py
python -m http.server --directory web 8000
```

Open http://localhost:8000 . The build step only copies release assets; it does
not require PyTorch or ONNX Runtime in Python. The browser downloads Pyodide,
NumPy and ONNX Runtime Web from pinned public CDNs.

Voice **92** is the recommended default. The example buttons use the sea story and four narrative sentences (Morning, Garden, Journey, Evening), screened for word accuracy. The browser keeps sentence context together where it fits. Speech naturalness can still vary; ASR correctness is not a human quality rating.
