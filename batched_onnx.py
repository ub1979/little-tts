"""Batched FP32 Little TTS inference on CPU or CUDA, without PyTorch."""
import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import onnxruntime as ort


class BatchedSynthesizer:
    def __init__(self, directory, device="cuda", device_id=0, max_batch=16,
                 max_padded_frames=16000, threads=2, profile=False, conv_pad_nc1d=True,
                 graph_subdir="onnx/batched"):
        self.directory = Path(directory).resolve()
        graph_dir = self.directory / graph_subdir
        self.metadata = json.loads((graph_dir / "config.json").read_text())
        if self.metadata.get("format") != "little-tts-batched-fp32-v1" or self.metadata.get("precision") != "float32" or self.metadata.get("quantized"):
            raise ValueError("Unsupported batched model format or precision")
        self.voices = self.metadata["voices"]
        self.sample_rate = self.metadata["sample_rate"]
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if max_batch < 1 or max_padded_frames < self.metadata["max_frames"]:
            raise ValueError("max_batch must be positive and frame budget must allow one maximum-length request")
        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable. Install onnxruntime-gpu and matching CUDA/cuDNN; CPU fallback is not used for a CUDA request.")
        if device == "cuda":
            # Load pip-provided NVIDIA libraries directly, without importing a training framework.
            ort.preload_dlls(directory="")
        self.device, self.device_id = device, device_id
        self.max_batch, self.max_padded_frames = max_batch, max_padded_frames
        providers = ["CPUExecutionProvider"] if device == "cpu" else [
            ("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0,
                                      "arena_extend_strategy": "kSameAsRequested",
                                      "cudnn_conv_algo_search": "DEFAULT",
                                      "cudnn_conv1d_pad_to_nc1d": int(conv_pad_nc1d),
                                      "cudnn_conv_use_max_workspace": "0"}),
            "CPUExecutionProvider"]
        def session(name):
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.enable_profiling = profile
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            result = ort.InferenceSession(str(graph_dir / name), options, providers=providers)
            if device == "cuda" and result.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA session initialization fell back to CPU")
            return result
        self.encoder, self.decoder = session("encoder.onnx"), session("decoder.onnx")
        from dizione.engine import Phonemizer
        self.phonemizer = Phonemizer(self.directory / "phonemizer")
        from dizione.symbols import SYMBOLS, tokenize
        if list(SYMBOLS) != self.metadata["symbols"]:
            raise ValueError("Pronunciation inventory does not match this model")
        self.tokenize = tokenize
        self.last_stats = {}
        self._positions_cache = {}
        self._run_lock = threading.Lock()

    def _value(self, array):
        return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), self.device, self.device_id)

    def _positions(self, kind, size):
        key = kind, size
        if key not in self._positions_cache:
            # Bound the cache across arbitrarily varied serving requests.
            if len(self._positions_cache) >= 32:
                self._positions_cache.clear()
            self._positions_cache[key] = self._value(np.arange(size, dtype=np.int64))
        return self._positions_cache[key]

    def _run(self, session, feeds, gpu_outputs=()):
        binding = session.io_binding()
        for key, value in feeds.items():
            binding.bind_ortvalue_input(key, value if isinstance(value, ort.OrtValue) else self._value(value))
        names = [x.name for x in session.get_outputs()]
        for name in names:
            binding.bind_output(name, self.device if name in gpu_outputs else "cpu", self.device_id)
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        outputs = binding.get_outputs()
        return {name: value if name in gpu_outputs else value.numpy() for name, value in zip(names, outputs)}

    def _infer_group(self, requests, result, batches):
        count = len(requests)
        n = max(len(r["phones"].ids) for r in requests)
        tokens, stress = np.zeros((count, n), np.int64), np.zeros((count, n), np.int64)
        mask = np.zeros((count, n), bool)
        for j, r in enumerate(requests):
            length = len(r["phones"].ids)
            tokens[j, :length], stress[j, :length], mask[j, :length] = r["phones"].ids, r["phones"].stress, True
        tokens_value = self._value(tokens)
        voice_value = self._value(np.array([r["voice"] for r in requests], np.int64))
        began = time.perf_counter()
        encoded = self._run(self.encoder, {
            "tokens": tokens_value, "stress": stress, "token_mask": mask,
            "token_positions": self._positions("tokens", n), "voice": voice_value,
            "speed": np.array([[r["speed"]] for r in requests], np.float32)}, gpu_outputs=("hidden",))
        encoder_seconds = time.perf_counter() - began
        durations = encoded["durations"]
        lengths = durations.sum(1)
        if np.any(lengths <= 0) or np.any(lengths > self.metadata["max_frames"]):
            raise ValueError("Predicted audio exceeds the model frame limit; split the text into shorter passages")
        frames = int(lengths.max())
        if count * frames > self.max_padded_frames and count > 1:
            # Split before allocating DSP tensors; no large hidden tensor is copied back to CPU.
            middle = count // 2
            self._infer_group(requests[:middle], result, batches)
            self._infer_group(requests[middle:], result, batches)
            return
        samples = frames * self.metadata["hop"]
        noise = np.zeros((count, samples), np.float32)
        for j, (r, length) in enumerate(zip(requests, lengths)):
            noise[j, :int(length)*256] = np.random.default_rng(r["seed"]).standard_normal(int(length)*256, dtype=np.float32)
        began = time.perf_counter()
        decoded = self._run(self.decoder, {
            "tokens": tokens_value, "hidden": encoded["hidden"], "durations": durations,
            "voice": voice_value, "frame_positions": self._positions("frames", frames),
            "sample_positions": self._positions("samples", samples),
            "noise_positions": self._positions("noise", frames*4+3), "noise": noise,
            "pitch_shift": np.array([[r["pitch_shift"]] for r in requests], np.float32),
            "pitch_range": np.array([[r["pitch_range"]] for r in requests], np.float32)})
        decoder_seconds = time.perf_counter() - began
        for j, r in enumerate(requests):
            size = int(decoded["sample_lengths"][j])
            wave = decoded["audio"][j, :size].copy()
            if not np.isfinite(wave).all():
                raise RuntimeError("Synthesis produced non-finite audio")
            result[r["index"]] = (wave, durations[j, :len(r["phones"].ids)].copy())
        batches.append(dict(batch_size=count, token_width=n, padded_frames=frames,
                            frame_lengths=lengths.tolist(), encoder_seconds=encoder_seconds,
                            decoder_seconds=decoder_seconds))

    def speak_batch(self, texts, voices=None, speeds=None, seeds=None, pitch_shifts=None, pitch_ranges=None):
        """Return [(waveform, durations), ...] in input order. CUDA is strict, not an automatic CPU fallback."""
        with self._run_lock:
            return self._speak_batch(texts, voices, speeds, seeds, pitch_shifts, pitch_ranges)

    def _speak_batch(self, texts, voices, speeds, seeds, pitch_shifts, pitch_ranges):
        texts = list(texts)
        if not texts:
            return []
        count = len(texts)
        def values(value, default):
            out = [default]*count if value is None else list(value)
            if len(out) != count: raise ValueError("Per-request settings must have the same length as texts")
            return out
        voices, speeds, seeds = values(voices, "92"), values(speeds, 1.0), values(seeds, 0)
        shifts, ranges = values(pitch_shifts, 0.0), values(pitch_ranges, 1.0)
        started = time.perf_counter()
        requests = []
        for j, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip(): raise ValueError("Each request needs nonempty text")
            if voices[j] not in self.voices: raise ValueError(f"voice must be one of {self.voices}")
            if not math.isfinite(speeds[j]) or not .5 <= speeds[j] <= 2: raise ValueError("speed must be between .5 and 2")
            if not math.isfinite(shifts[j]) or not math.isfinite(ranges[j]) or not 0 <= ranges[j] <= 3: raise ValueError("Pitch settings must be finite; pitch range must be between 0 and 3")
            if not isinstance(seeds[j], (int, np.integer)) or seeds[j] < 0: raise ValueError("Seeds must be nonnegative integers")
            phones = self.tokenize(self.phonemizer.pieces(text))
            if len(phones.ids) > self.metadata["max_tokens"]: raise ValueError("Text exceeds the phoneme limit; split it into shorter passages")
            requests.append(dict(index=j, phones=phones, voice=self.voices.index(voices[j]), speed=speeds[j],
                                 seed=seeds[j], pitch_shift=shifts[j], pitch_range=ranges[j]))
        tokenization_seconds = time.perf_counter() - started
        # Similar token lengths usually imply similar audio lengths, limiting padding cost.
        requests.sort(key=lambda r: len(r["phones"].ids))
        result, batches = [None]*count, []
        for first in range(0, count, self.max_batch):
            self._infer_group(requests[first:first+self.max_batch], result, batches)
        elapsed = time.perf_counter() - started
        total_audio = sum(len(wave) for wave, _ in result) / self.sample_rate
        self.last_stats = dict(device=self.device, requests=count, total_seconds=elapsed,
                               tokenization_seconds=tokenization_seconds, total_audio_seconds=total_audio,
                               real_time_factor=elapsed/total_audio, batches=batches)
        return result

    def end_profiling(self):
        return [self.encoder.end_profiling(), self.decoder.end_profiling()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--text", action="append", required=True)
    p.add_argument("--voice", default="92")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path("batch-audio"))
    args = p.parse_args()
    import soundfile as sf
    synth = BatchedSynthesizer(args.model_dir, args.device, max_batch=args.batch_size)
    result = synth.speak_batch(args.text, voices=[args.voice]*len(args.text))
    args.output.mkdir(parents=True, exist_ok=True)
    for j, (wave, _) in enumerate(result): sf.write(args.output/f"{j:03d}.wav", wave, synth.sample_rate, subtype="FLOAT")
    print(json.dumps(synth.last_stats, indent=2))


if __name__ == "__main__":
    main()
