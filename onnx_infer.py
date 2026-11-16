"""English DDSP inference using NumPy + ONNX Runtime, with no PyTorch import."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Synthesizer:
    def __init__(self, directory, threads=2, model_file="model.onnx"):
        self.directory = Path(directory).resolve()
        os.environ["DIZIONE_DATA"] = str(self.directory / "phonemizer")
        import dizione
        dizione.reset()
        self.phonemizer = dizione
        self.metadata = json.loads((self.directory / "onnx/model.json").read_text())
        self.voices = self.metadata["voices"]
        self.sample_rate = self.metadata["sample_rate"]
        if tuple(dizione.symbols.SYMBOLS) != tuple(self.metadata["symbols"]):
            raise ValueError("Phoneme inventory does not match the exported model")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.directory / "onnx" / model_file), options,
                                            providers=["CPUExecutionProvider"])

    def speak(self, text, voice="92", speed=1.0):
        if voice not in self.voices:
            raise ValueError(f"voice must be one of {self.voices}")
        if not 0.5 <= float(speed) <= 2:
            raise ValueError("speed must be between 0.5 and 2")
        phones = self.phonemizer.phonemize(text)
        if len(phones.ids) > self.metadata["max_tokens"]:
            raise ValueError("Text is too long; split it into sentences")
        one = lambda x: np.array([x], dtype=np.float32)
        wave, durations = self.session.run(None, {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([self.voices.index(voice)], dtype=np.int64),
            "speed": one(speed), "pitch_shift": one(0), "pitch_range": one(1),
        })
        if not np.isfinite(wave).all():
            raise RuntimeError("Synthesis produced non-finite audio")
        return wave[0], durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="92")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("speech.wav"))
    args = parser.parse_args()
    import soundfile as sf
    synth = Synthesizer(args.model_dir, args.threads)
    wave, _ = synth.speak(args.text, args.voice, args.speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wave, synth.sample_rate, subtype="FLOAT")
    print(f"Wrote {len(wave)/synth.sample_rate:.2f}s to {args.output}")


if __name__ == "__main__":
    main()
