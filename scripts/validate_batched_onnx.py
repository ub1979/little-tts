"""Compare batched FP32 audio with the released single-request ONNX graph."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batched_onnx import BatchedSynthesizer

TEXTS = [
    "The morning sunlight streamed through the window, warming the quiet room.",
    "Hello, welcome to Little TTS.",
    "She had never seen the sea before, and for a long while she stood at the edge of the water without saying a word.",
    "After a long and difficult journey across the mountains, the travellers finally reached the small village, where they were welcomed with warm food, a place to sleep and many questions about the road behind them.",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=ROOT)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, default=ROOT / "evaluation/batched_validation.json")
    args = p.parse_args()
    synth = BatchedSynthesizer(args.model_dir, args.device, max_batch=4)
    # Expose only the original graph's random noise, leaving all learned operators unchanged.
    model = onnx.load(args.model_dir / "onnx/model.onnx")
    for node in model.graph.node:
        if node.op_type == "RandomNormalLike":
            del node.input[:]
            node.input.append("reference_noise")
            node.op_type = "Identity"
            del node.attribute[:]
    model.graph.input.append(h.make_tensor_value_info("reference_noise", T.FLOAT, [1, "samples"]))
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "reference.onnx"
        onnx.save(model, path)
        reference = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        rows = []
        cases = [(["92", "6097", "92", "6097"], [1, 1.1, .95, 1], [0]*4, [1]*4),
                 (["6097", "92", "6097", "92"], [1.1, 1, 1, .95], [0]*4, [1]*4),
                 (["92", "6097", "92", "6097"], [1]*4, [.05, -.03, .02, 0], [1.2, .8, 1.1, 1])]
        for voices, speeds, shifts, ranges in cases:
            result = synth.speak_batch(TEXTS, voices=voices, speeds=speeds, seeds=[10, 11, 12, 13],
                                       pitch_shifts=shifts, pitch_ranges=ranges)
            for j, ((wave, durations), text, voice, speed) in enumerate(zip(result, TEXTS, voices, speeds)):
                phones = synth.tokenize(synth.phonemizer.pieces(text))
                noise = np.random.default_rng(10+j).standard_normal(len(wave), dtype=np.float32)[None]
                one = lambda v: np.array([v], np.float32)
                feeds = dict(tokens=np.array([phones.ids], np.int64), stress=np.array([phones.stress], np.int64),
                             voice=np.array([synth.voices.index(voice)], np.int64), speed=one(speed),
                             pitch_shift=one(shifts[j]), pitch_range=one(ranges[j]), reference_noise=noise)
                original, original_durations = reference.run(None, feeds)
                assert durations.tolist() == original_durations[0].tolist(), (j, "durations differ")
                assert len(wave) == original.shape[1], (j, "length differs")
                original = original[0]
                peak = float(np.max(np.abs(original)))
                error = wave - original
                row = dict(voice=voice, text=text, speed=speed, pitch_shift=shifts[j], pitch_range=ranges[j], seconds=len(wave)/24000,
                           same_durations=True, same_length=True,
                           max_error_over_peak=float(np.max(np.abs(error))/peak),
                           rms_error_over_rms=float(np.sqrt(np.mean(error**2))/np.sqrt(np.mean(original**2))))
                rows.append(row)
                print(json.dumps(row), flush=True)
        # Batch invariance, mixed settings and input ordering are independently checked.
        batch = synth.speak_batch(TEXTS[:3], voices=["6097", "92", "6097"], seeds=[5, 6, 7])
        invariance = []
        for j in range(3):
            single = synth.speak_batch([TEXTS[j]], voices=[["6097", "92", "6097"][j]], seeds=[5+j])[0]
            assert np.array_equal(batch[j][1], single[1])
            delta = np.max(np.abs(batch[j][0] - single[0]))
            peak = np.max(np.abs(single[0]))
            invariance.append(float(delta/peak))
        # Force a padded-frame-budget split and verify output ordering/durations.
        limited = BatchedSynthesizer(args.model_dir, args.device, max_batch=4, max_padded_frames=3000)
        limited_result = limited.speak_batch(TEXTS, seeds=[10, 11, 12, 13])
        assert len(limited.last_stats["batches"]) > 1
        assert all(b["batch_size"] * b["padded_frames"] <= 3000 for b in limited.last_stats["batches"])
        for j, (wave, durations) in enumerate(limited_result):
            single = synth.speak_batch([TEXTS[j]], seeds=[10+j])[0]
            assert np.array_equal(durations, single[1])
            assert np.max(np.abs(wave - single[0])) / np.max(np.abs(single[0])) < .005
        report = dict(device=args.device, precision="float32", quantized=False, pytorch_imported="torch" in sys.modules,
                      reference="Released FP32 single-request ONNX, with identical externally supplied noise",
                      comparisons=rows, batch_invariance_max_error_over_peak=invariance,
                      padded_frame_budget_checked=True,
                      max_error_over_peak=max(r["max_error_over_peak"] for r in rows),
                      max_rms_error_over_rms=max(r["rms_error_over_rms"] for r in rows))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert not report["pytorch_imported"]
        assert report["max_error_over_peak"] < .005, report
        assert report["max_rms_error_over_rms"] < .002, report
        assert max(invariance) < .005, report
        print("BATCHED_PARITY_PASSED", flush=True)


if __name__ == "__main__":
    main()
