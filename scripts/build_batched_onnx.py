"""Build FP32 batched inference graphs directly from the released ONNX weights.

No training framework is imported. Two stages expose the duration-dependent
shape boundary to the host; large hidden tensors can remain on CUDA via I/O
binding. The original browser model is not modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.model = onnx.load(path)
        self.nodes = {n.name: n for n in self.model.graph.node}
        self.values = {t.name: nh.to_array(t) for t in self.model.graph.initializer}
        for n in self.model.graph.node:
            if n.op_type == "Constant":
                self.values[n.output[0]] = nh.to_array(next(a.t for a in n.attribute if a.name == "value"))

    def constant(self, name):
        return self.values[self.nodes[name].output[0]]


class Graph:
    def __init__(self, source):
        self.source, self.nodes, self.weights = source, [], {}
        self.counter = 0

    def c(self, value, dtype=None, name=None):
        value = np.asarray(value, dtype=dtype)
        key = name or "constant_" + hashlib.sha256(value.tobytes() + str(value.shape).encode() + value.dtype.str.encode()).hexdigest()[:18]
        self.weights.setdefault(key, nh.from_array(value, key))
        return key

    def op(self, kind, *inputs, **attrs):
        self.counter += 1
        name = f"{kind}_{self.counter}"
        self.nodes.append(h.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def f(self, value): return self.c(value, np.float32)
    def i(self, value): return self.c(value, np.int64)
    def unsq(self, x, axes): return self.op("Unsqueeze", x, self.i(axes))
    def cast(self, x, dtype=T.FLOAT): return self.op("Cast", x, to=dtype)
    def add(self, x, y): return self.op("Add", x, y)
    def sub(self, x, y): return self.op("Sub", x, y)
    def mul(self, x, y): return self.op("Mul", x, y)
    def div(self, x, y): return self.op("Div", x, y)
    def clip(self, x, low, high): return self.op("Clip", x, self.f(low), self.f(high))
    def reshape(self, x, shape): return self.op("Reshape", x, self.i(shape))
    def cat(self, *xs, axis=-1): return self.op("Concat", *xs, axis=axis)
    def sum(self, x, axes, keep=1): return self.op("ReduceSum", x, self.i(axes), keepdims=keep)
    def sl(self, x, start, end, axis=-1, step=1):
        return self.op("Slice", x, self.i([start]), self.i([end]), self.i([axis]), self.i([step]))

    def named(self, name):
        return self.c(self.source.values[name], name=name)

    def linear(self, x, path, module=None):
        weight = self.source.nodes[f"/{path}/MatMul"].input[1]
        bias = "model." + (module or path.replace("/", ".")) + ".bias"
        return self.add(self.op("MatMul", x, self.named(weight)), self.named(bias))

    def norm(self, x, module):
        return self.op("LayerNormalization", x, self.named("model." + module + ".weight"),
                       self.named("model." + module + ".bias"), axis=-1, epsilon=1e-5)

    def gelu(self, x):
        return self.mul(self.mul(x, self.add(self.op("Erf", self.div(x, self.f(math.sqrt(2)))), self.f(1))), self.f(.5))

    def conv(self, x, module, groups=1, kernel=3):
        return self.op("Conv", x, self.named("model." + module + ".weight"), self.named("model." + module + ".bias"),
                       group=groups, kernel_shape=[kernel], pads=[kernel//2, kernel//2])

    def rotary(self, x, cos, sin):
        even, odd = self.sl(x, 0, 48, step=2), self.sl(x, 1, 48, step=2)
        a = self.sub(self.mul(even, cos), self.mul(odd, sin))
        b = self.add(self.mul(even, sin), self.mul(odd, cos))
        return self.reshape(self.cat(self.unsq(a, [-1]), self.unsq(b, [-1])), [0, 0, 0, 48])

    def stack(self, x, mask, positions, module, layers):
        keep = self.unsq(self.cast(mask), [-1])
        angle = self.mul(self.unsq(self.cast(positions), [-1]), self.c(self.source.constant("/encoder/Constant_4")))
        cos, sin = self.op("Cos", angle), self.op("Sin", angle)
        attention_mask = self.unsq(mask, [1, 2])
        for j in range(layers):
            p = f"{module}/blocks.{j}"
            key = p.replace("/", ".")
            y = self.norm(x, key + ".norm1")
            qkv = self.reshape(self.linear(y, p + "/attention/qkv"), [0, 0, 3, 4, 48])
            q, k, v = [self.op("Transpose", self.op("Gather", qkv, self.i(n), axis=2), perm=[0, 2, 1, 3]) for n in range(3)]
            scale = self.f(48 ** -.25)
            q = self.mul(self.rotary(q, cos, sin), scale)
            k = self.mul(self.rotary(k, cos, sin), scale)
            scores = self.op("MatMul", q, self.op("Transpose", k, perm=[0, 1, 3, 2]))
            scores = self.op("Where", attention_mask, scores, self.f(-np.inf))
            y = self.op("MatMul", self.op("Softmax", scores, axis=-1), v)
            y = self.reshape(self.op("Transpose", y, perm=[0, 2, 1, 3]), [0, 0, 192])
            x = self.mul(self.add(x, self.linear(y, p + "/attention/out")), keep)
            y = self.mul(self.norm(x, key + ".norm2"), keep)
            y = self.op("Transpose", y, perm=[0, 2, 1])
            y = self.op("Transpose", self.conv(y, key + ".depthwise", 192, 7), perm=[0, 2, 1])
            y = self.linear(self.gelu(self.linear(y, p + "/expand")), p + "/project")
            x = self.mul(self.add(x, y), keep)
        return self.mul(self.norm(x, module + ".norm"), keep)

    def take(self, x, indices, channels):
        shape = self.cat(self.op("Shape", indices), self.i([channels]), axis=0)
        expanded = self.op("Expand", self.unsq(indices, [-1]), shape)
        return self.op("GatherElements", x, expanded, axis=1)

    def exp_sigmoid(self, x, maximum=2):
        return self.add(self.mul(self.f(maximum), self.op("Pow", self.op("Sigmoid", x), self.f(math.log(10)))), self.f(1e-7))

    def save(self, path, inputs, outputs):
        for value, info in outputs:
            self.nodes.append(h.make_node("Identity", [value], [info.name], name="output_" + info.name))
        graph = h.make_graph(self.nodes, path.stem, inputs, [v for _, v in outputs], list(self.weights.values()))
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], producer_name="Little TTS FP32 batched ONNX builder")
        model.ir_version = 10
        onnx.checker.check_model(model)
        onnx.save(model, path)
        return len(self.nodes)


def vi(name, dtype, shape): return h.make_tensor_value_info(name, dtype, shape)


def masks(source):
    arrays = [a for a in source.values.values() if a.dtype == np.bool_ and a.shape == (76,)]
    return max(arrays, key=np.count_nonzero), min(arrays, key=np.count_nonzero)


def build_encoder(source, path):
    g = Graph(source)
    phone_mask, _ = masks(source)
    embed = lambda key, index: g.op("Gather", g.named("model." + key + ".weight"), index, axis=0)
    x = g.add(g.add(embed("tokens", "tokens"), embed("stress", "stress")), g.unsq(embed("voices", "voice"), [1]))
    hidden = g.stack(x, "token_mask", "token_positions", "encoder", 4)
    keep = g.unsq(g.cast("token_mask"), [-1])
    x = g.op("Transpose", g.mul(hidden, keep), perm=[0, 2, 1])
    x = g.op("Transpose", g.conv(x, "duration.conv1"), perm=[0, 2, 1])
    x = g.mul(g.norm(g.gelu(x), "duration.norm1"), keep)
    x = g.op("Transpose", g.conv(g.op("Transpose", x, perm=[0, 2, 1]), "duration.conv2"), perm=[0, 2, 1])
    x = g.mul(g.linear(g.norm(g.gelu(x), "duration.norm2"), "duration/out"), keep)
    raw = g.op("Gather", x, g.i(0), axis=2)
    frames = g.cast(g.op("Round", g.div(g.sub(g.op("Exp", g.clip(raw, 0, math.log1p(400))), g.f(1)), "speed")), T.INT64)
    frames = g.op("Max", frames, g.i(1))
    present = g.op("Or", g.op("Gather", g.c(phone_mask), "tokens", axis=0), g.op("Greater", g.op("Gather", x, g.i(1), axis=2), g.f(0)))
    present = g.op("And", present, "token_mask")
    durations = g.op("Where", present, frames, g.i(0))
    return g.save(path, [vi("tokens", T.INT64, ["B", "N"]), vi("stress", T.INT64, ["B", "N"]),
                        vi("token_mask", T.BOOL, ["B", "N"]), vi("token_positions", T.INT64, ["N"]),
                        vi("voice", T.INT64, ["B"]), vi("speed", T.FLOAT, ["B", 1])],
                  [(hidden, vi("hidden", T.FLOAT, ["B", "N", 192])), (durations, vi("durations", T.INT64, ["B", "N"]))])


def build_decoder(source, path, tile=16, reduction="matmul", noise_filter="matmul"):
    g = Graph(source)
    _, vowels = masks(source)
    lengths = g.sum("durations", [1], keep=0)
    ends = g.op("CumSum", "durations", g.i(1))
    frame = g.unsq("frame_positions", [0])
    mask = g.op("Less", frame, g.unsq(lengths, [1]))
    index = g.sum(g.cast(g.op("LessOrEqual", g.unsq(ends, [1]), g.unsq(frame, [-1])), T.INT64), [-1], keep=0)
    n = g.op("Gather", g.op("Shape", "durations"), g.i(1), axis=0)
    index = g.op("Min", index, g.sub(n, g.i(1)))
    owned = g.op("Max", g.op("GatherElements", "durations", index, axis=1), g.i(1))
    offset = g.sub(frame, g.op("GatherElements", g.sub(ends, "durations"), index, axis=1))
    relative = g.div(g.add(g.cast(offset), g.f(.5)), g.cast(owned))
    features = g.cat(g.unsq(relative, [-1]), g.unsq(g.sub(g.f(1), relative), [-1]),
                     g.unsq(g.div(g.op("Log", g.add(g.cast(owned), g.f(1))), g.f(4)), [-1]))
    expanded = g.add(g.take("hidden", index, 192), g.linear(features, "frame_position"))
    expanded = g.add(expanded, g.unsq(g.op("Gather", g.named("model.frame_voices.weight"), "voice", axis=0), [1]))
    prosody_hidden = g.stack(expanded, mask, "frame_positions", "prosody_stack", 2)
    prosody = g.linear(prosody_hidden, "prosody_head")
    component = lambda i: g.op("Gather", prosody, g.i(i), axis=2)
    vowel = g.op("GatherElements", g.op("Gather", g.c(vowels), "tokens", axis=0), index, axis=1)
    voicing = g.op("Where", vowel, g.op("Max", component(1), g.f(2)), component(1))
    voiced = g.mul(g.cast(g.op("Greater", voicing, g.f(0))), g.cast(mask))
    mean = g.div(g.sum(g.mul(component(0), voiced), [1]), g.op("Max", g.sum(voiced, [1]), g.f(1)))
    pitch = g.add(mean, g.mul(g.sub(component(0), mean), "pitch_range"))
    condition = g.cat(g.unsq(g.clip(pitch, -2, 1.5), [-1]), g.unsq(voiced, [-1]), g.unsq(g.clip(component(2), -4.5, 3), [-1]))
    acoustic = g.stack(g.add(prosody_hidden, g.linear(condition, "prosody_projection")), mask, "frame_positions", "acoustic_stack", 5)
    f0 = g.clip(g.mul(g.f(200), g.op("Exp", g.add(pitch, "pitch_shift"))), 50, 600)
    # Repeat each request's own last real frame at its boundary, never a neighbour's padding.
    here = g.op("Min", frame, g.sub(g.unsq(lengths, [1]), g.i(1)))
    following = g.op("Min", g.add(frame, g.i(1)), g.sub(g.unsq(lengths, [1]), g.i(1)))
    acoustic = g.take(acoustic, here, 192)
    f0 = g.op("GatherElements", f0, here, axis=1)
    x = g.add(acoustic, g.linear(g.gelu(g.norm(g.linear(acoustic, "mlp_in", "vocoder.mlp_in"), "vocoder.mlp_norm")), "mlp_out", "vocoder.mlp_out"))
    harmonic = g.linear(x, "harmonic", "vocoder.harmonic")
    amp = g.exp_sigmoid(g.sl(harmonic, 0, 1), 1)
    distribution = g.exp_sigmoid(g.sl(harmonic, 1, 65))
    phi = g.mul(g.f(math.pi), g.op("Tanh", g.sl(harmonic, 65, 129)))
    harmonics = g.f(np.arange(1, 65, dtype=np.float32))
    frequency = g.unsq(f0, [-1])
    nyquist = g.clip(g.div(g.sub(g.f(.47 * 24000), g.mul(frequency, harmonics)), g.op("Max", frequency, g.f(1))), 0, 1)
    distribution = g.mul(distribution, nyquist)
    distribution = g.div(distribution, g.op("Max", g.sum(distribution, [-1]), g.f(1e-6)))
    amplitudes = g.mul(amp, distribution)
    a, b = g.mul(amplitudes, g.op("Cos", phi)), g.mul(amplitudes, g.op("Sin", phi))
    ramp = g.f(np.arange(256, dtype=np.float32) / 256)
    knots = g.op("Log", g.op("Max", f0, g.f(1)))
    next_knots = g.op("GatherElements", knots, following, axis=1)
    sample_f0 = g.op("Exp", g.add(g.mul(g.unsq(knots, [-1]), g.sub(g.f(1), ramp)), g.mul(g.unsq(next_knots, [-1]), ramp)))
    within = g.op("CumSum", g.div(sample_f0, g.f(24000)), g.i(2))
    whole = g.op("Gather", within, g.i(255), axis=2)
    steps = g.cast(g.op("Round", g.mul(g.sub(whole, g.op("Floor", whole)), g.f(1 << 24))), T.INT64)
    start = g.div(g.cast(g.op("Mod", g.sub(g.op("CumSum", steps, g.i(1)), steps), g.i(1 << 24), fmod=0)), g.f(1 << 24))
    cycles = g.add(within, g.unsq(start, [-1]))
    theta = g.mul(g.f(2 * math.pi), g.sub(cycles, g.op("Floor", cycles)))
    wave = None
    for k in range(0, 64, tile):
        hi = min(k + tile, 64)
        a0, b0 = g.sl(a, k, hi), g.sl(b, k, hi)
        a1, b1 = g.take(a0, following, hi-k), g.take(b0, following, hi-k)
        angle = g.mul(g.unsq(theta, [-1]), g.f(np.arange(k+1, hi+1, dtype=np.float32)))
        if reduction == "matmul":
            low_high_a = g.cat(g.unsq(a0, [-1]), g.unsq(a1, [-1]))
            low_high_b = g.cat(g.unsq(b0, [-1]), g.unsq(b1, [-1]))
            pair = g.add(g.op("MatMul", g.op("Sin", angle), low_high_a), g.op("MatMul", g.op("Cos", angle), low_high_b))
            blend = g.add(g.mul(g.op("Gather", pair, g.i(0), axis=3), g.sub(g.f(1), ramp)), g.mul(g.op("Gather", pair, g.i(1), axis=3), ramp))
        else:
            # Avoid thousands of tiny strided GEMMs in the oscillator bank.
            r = g.unsq(ramp, [-1])
            mix_a = g.add(g.mul(g.unsq(a0, [2]), g.sub(g.f(1), r)), g.mul(g.unsq(a1, [2]), r))
            mix_b = g.add(g.mul(g.unsq(b0, [2]), g.sub(g.f(1), r)), g.mul(g.unsq(b1, [2]), r))
            blend = g.sum(g.add(g.mul(g.op("Sin", angle), mix_a), g.mul(g.op("Cos", angle), mix_b)), [-1], keep=0)
        wave = blend if wave is None else g.add(wave, blend)
    wave = g.reshape(wave, [0, -1])
    sample_lengths = g.mul(lengths, g.i(256))
    audible = g.op("Less", g.unsq("sample_positions", [0]), g.unsq(sample_lengths, [1]))
    # Noise filtering is block-diagonal across the batch and zero-padded per request.
    noise_head = g.linear(x, "noise", "vocoder.noise")
    sub = g.op("Min", g.unsq("noise_positions", [0]), g.sub(g.mul(g.unsq(lengths, [1]), g.i(4)), g.i(1)))
    frame_sub = g.div(sub, g.i(4))
    bands = g.exp_sigmoid(g.take(g.sl(noise_head, 0, 65), frame_sub, 65))
    gain = g.op("GatherElements", g.reshape(g.sl(noise_head, 65, 69), [0, -1]), sub, axis=1)
    response = g.op("MatMul", g.mul(bands, g.unsq(g.exp_sigmoid(gain, 1), [-1])), g.c(source.constant("/Constant_137")))
    source_noise = g.mul("noise", g.cast(audible))
    original_convs = [n for n in source.model.graph.node if n.op_type == "ConvTranspose"]
    synthesis = source.values[original_convs[0].input[1]]
    envelope_kernel = source.values[original_convs[1].input[1]]
    if noise_filter == "conv":
        padded = g.op("Pad", g.unsq(source_noise, [1]), g.i([0, 0, 128, 0, 0, 256]), mode="constant")
        real = g.op("Conv", padded, g.c(source.constant("/Constant_148")), strides=[64])
        imag = g.op("Conv", padded, g.c(source.constant("/Constant_149")), strides=[64])
        response = g.op("Transpose", response, perm=[0, 2, 1])
        product = g.cat(g.mul(real, response), g.mul(imag, response), axis=1)
        noise_wave = g.op("ConvTranspose", product, g.c(synthesis), strides=[64])
        ones = g.add(g.mul(g.sl(real, 0, 1, axis=1), g.f(0)), g.f(1))
        envelope = g.op("ConvTranspose", ones, g.c(envelope_kernel), strides=[64])
        noise_wave = g.op("Gather", g.div(noise_wave, envelope), g.i(0), axis=1)
    else:
        # Exact real-valued STFT matrices, as in the original export, using large
        # GEMMs rather than small 1D convolution / transposed-convolution kernels.
        padded = g.op("Pad", source_noise, g.i([0, 128, 0, 256]), mode="constant")
        indices = g.add(g.mul(g.unsq("noise_positions", [1]), g.i(64)), g.i(np.arange(256)))
        frames = g.op("Gather", padded, indices, axis=1)
        analysis = np.concatenate([source.constant("/Constant_148")[:, 0, :].T,
                                   source.constant("/Constant_149")[:, 0, :].T], axis=1)
        spectrum = g.op("MatMul", frames, g.c(analysis))
        product = g.mul(spectrum, g.cat(response, response))
        filtered_frames = g.op("MatMul", product, g.c(synthesis[:, 0, :]))

        def overlap_add(values):
            blocks = g.reshape(values, [0, 0, 4, 64])
            pieces = []
            for k in range(4):
                piece = g.op("Gather", blocks, g.i(k), axis=2)
                pieces.append(g.op("Pad", piece, g.i([0, k, 0, 0, 3-k, 0]), mode="constant"))
            total = pieces[0]
            for piece in pieces[1:]: total = g.add(total, piece)
            return g.reshape(total, [0, -1])

        noise_wave = overlap_add(filtered_frames)
        ones = g.add(g.mul(g.cast("noise_positions"), g.f(0)), g.f(1))
        envelopes = g.mul(g.unsq(ones, [0, 2]), g.c(envelope_kernel[0, 0]))
        envelope = overlap_add(envelopes)
        noise_wave = g.div(noise_wave, envelope)
    noise_wave = g.op("Gather", noise_wave, g.add("sample_positions", g.i(128)), axis=1)
    audio = g.mul(g.add(wave, noise_wave), g.cast(audible))
    inputs = [vi("tokens", T.INT64, ["B", "N"]), vi("hidden", T.FLOAT, ["B", "N", 192]),
              vi("durations", T.INT64, ["B", "N"]), vi("voice", T.INT64, ["B"]),
              vi("frame_positions", T.INT64, ["F"]), vi("sample_positions", T.INT64, ["S"]),
              vi("noise_positions", T.INT64, ["Q"]), vi("noise", T.FLOAT, ["B", "S"]),
              vi("pitch_shift", T.FLOAT, ["B", 1]), vi("pitch_range", T.FLOAT, ["B", 1])]
    return g.save(path, inputs, [(audio, vi("audio", T.FLOAT, ["B", "S"])), (sample_lengths, vi("sample_lengths", T.INT64, ["B"]))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--source", type=Path, default=root / "onnx/model.onnx")
    p.add_argument("--output", type=Path, default=root / "onnx/batched")
    p.add_argument("--harmonic-tile", type=int, choices=[8, 16, 32, 64], default=16)
    p.add_argument("--harmonic-reduction", choices=["sum", "matmul"], default="matmul")
    p.add_argument("--noise-filter", choices=["matmul", "conv"], default="matmul")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Source(args.source)
    counts = {"encoder": build_encoder(source, args.output / "encoder.onnx"),
              "decoder": build_decoder(source, args.output / "decoder.onnx", args.harmonic_tile, args.harmonic_reduction, args.noise_filter)}
    metadata = json.loads(args.source.with_suffix(".json").read_text())
    metadata.pop("inputs", None)
    metadata.pop("outputs", None)
    metadata.update(format="little-tts-batched-fp32-v1", source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    harmonic_tile=args.harmonic_tile, harmonic_reduction=args.harmonic_reduction, noise_filter=args.noise_filter, graph_nodes=counts,
                    graphs=["encoder.onnx", "decoder.onnx"], precision="float32", quantized=False)
    metadata["graph_inputs"] = {
        "encoder.onnx": ["tokens", "stress", "token_mask", "token_positions", "voice", "speed"],
        "decoder.onnx": ["tokens", "hidden", "durations", "voice", "frame_positions", "sample_positions",
                         "noise_positions", "noise", "pitch_shift", "pitch_range"]}
    metadata["graph_outputs"] = {"encoder.onnx": ["hidden", "durations"], "decoder.onnx": ["audio", "sample_lengths"]}
    metadata["graph_sha256"] = {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in metadata["graphs"]}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(counts), "FP32 graphs built without PyTorch")


if __name__ == "__main__":
    main()
