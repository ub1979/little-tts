"""Clean real-human Hi-Fi TTS preparation, alignment and DDSP training.

Run after downloading MikhailT/hifi-tts clean shards at REVISION to
human_data/hifitts. The synthetic-data directories and weights are unused.
"""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASET = "MikhailT/hifi-tts"
REVISION = "f0a934ddf4e8da0fc6c53fb42fda2c9ac8ed93df"
ALIGNER_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
BASE = ROOT / "human_data/hifitts_run"
SOURCE = ROOT / "human_data/hifitts"
CONFIG = BASE / "hifi_config.json"
FEATURES = BASE / "features"
ANNOTATIONS = BASE / "annotations"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def train_with_previews(args, output):
    import torch
    samples = output.parent / (output.name + "_samples")
    samples.mkdir(parents=True, exist_ok=True)
    seen = -1
    process = subprocess.Popen(list(map(str, args)))
    try:
        while True:
            checkpoint = output / "inference.pt"
            if checkpoint.exists():
                shutil.copyfile(BASE / "DATA_LICENSE.txt", output / "DATA_LICENSE.txt")
                shutil.copyfile(BASE / "provenance.json", output / "data_provenance.json")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                step = state["step"]
                if step > seen:
                    temporary = samples / ".preview_checkpoint.pt"
                    torch.save(state, temporary)
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/render_training_samples.py"),
                        "--checkpoint", str(temporary), "--output", str(samples / f"step_{step:06d}")])
                    if result.returncode:
                        print("Sample generation failed at step", step, "training continues", flush=True)
                    else:
                        print("SAMPLES_READY", step, str(samples / f"step_{step:06d}"), flush=True)
                    temporary.unlink(missing_ok=True)
                    seen = step
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                continue
            # Make one final pass so the last saved checkpoint also gets previews.
            if checkpoint.exists() and torch.load(checkpoint, map_location="cpu", weights_only=True)["step"] > seen:
                continue
            if code:
                raise subprocess.CalledProcessError(code, args)
            break
    finally:
        if process.poll() is None:
            (ROOT / "STOP_HUMAN_TRAINING").touch()


def resample(item):
    import soundfile as sf
    import soxr
    import numpy as np
    identifier, audio = item
    target = BASE / "wavs_24k" / (identifier + ".wav")
    if not target.exists():
        wave, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if wave.ndim != 1 or rate != 44100:
            raise ValueError(f"Unexpected source audio format: {identifier}")
        if not np.isfinite(wave).all() or np.sqrt(np.mean(wave**2)) < 1e-5:
            return identifier, "silent_or_nonfinite"
        if np.mean(np.abs(wave) >= .999) > .001:
            return identifier, "clipped"
        temporary = target.with_suffix(".tmp.wav")
        output_wave = soxr.resample(wave, rate, 24000, quality="VHQ")
        peak = float(np.max(np.abs(output_wave)))
        if peak > .99:
            output_wave *= .99 / peak
        sf.write(temporary, output_wave, 24000, subtype="PCM_16")
        temporary.replace(target)
    return identifier, None


def dataset_rows(config):
    import pyarrow.parquet as pq
    catalog = {split: [] for split in ("train", "validation", "test")}
    sources = sorted((SOURCE / "data").glob("*.clean-*.parquet"))
    if len(sources) != 37:
        raise ValueError(f"Expected 37 clean dataset shards, found {len(sources)}")
    for source in sources:
        split = "validation" if source.name.startswith("dev.") else "test" if source.name.startswith("test.") else "train"
        for record in pq.read_table(source, columns=["speaker", "file", "duration", "text_normalized"]).to_pylist():
            if record["speaker"] not in config.speakers or not .8 <= record["duration"] <= 15:
                continue
            text = record["text_normalized"].strip()
            if len(text) < 5:
                continue
            identifier = "hifi_" + record["speaker"] + "_" + hashlib.sha256(record["file"].encode()).hexdigest()[:20]
            catalog[split].append(dict(id=identifier, split=split, voice_id=record["speaker"], text=text,
                audio=str(BASE / "wavs_24k" / (identifier + ".wav")), origin="Hi-Fi TTS clean real human recording",
                source_id=record["file"], source_license="CC BY 4.0",
                source_url=f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                duration=record["duration"]))
    # Keep roughly 15 hours per speaker, selected across books by stable hashes.
    chosen, hours = [], {s: 0. for s in config.speakers}
    held_out = {r["text"] for split in ("validation", "test") for r in catalog[split]}
    for row in sorted(catalog["train"], key=lambda r: r["id"]):
        speaker = row["voice_id"]
        if row["text"] in held_out or hours[speaker] + row["duration"] > 15 * 3600:
            continue
        chosen.append(row)
        hours[speaker] += row["duration"]
    catalog["train"] = chosen
    test_text = {r["text"] for r in catalog["test"]}
    catalog["validation"] = [r for r in catalog["validation"] if r["text"] not in test_text]
    wanted = {r["source_id"]: r for rows in catalog.values() for r in rows}
    if len(wanted) != sum(map(len, catalog.values())):
        raise ValueError("Duplicate recording across source splits")
    (BASE / "wavs_24k").mkdir(parents=True, exist_ok=True)
    pending = {path: row for path, row in wanted.items() if not Path(row["audio"]).exists()}
    rejected = {}
    count = len(wanted) - len(pending)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for source in sources:
            if not pending:
                break
            for batch in pq.ParquetFile(source).iter_batches(batch_size=16, columns=["file", "audio"]):
                items = [(pending[r["file"]]["id"], r["audio"]) for r in batch.to_pylist() if r["file"] in pending]
                for identifier, reason in pool.map(resample, items):
                    count += 1
                    if reason:
                        rejected[identifier] = reason
            print("Decoded and resampled real recordings:", count, "shard:", source.name, flush=True)
    for split in catalog:
        catalog[split] = [r for r in catalog[split] if r["id"] not in rejected]
    write_json(BASE / "audio_qc.json", dict(rejected=rejected, checks=["clean source subset", "mono 44100 Hz",
               "finite non-silent signal", "clipped samples <= 0.1%", "duration 0.8 to 15 seconds"]))
    return catalog


def prepare(task):
    split, manifest = task
    output = BASE / "feature_shards" / manifest.stem
    with (ROOT / "logs" / (manifest.stem + ".log")).open("a") as log:
        run(sys.executable, "-m", "little_tts.prepare", "--inputs", manifest,
            "--split", split, "--output", output, "--config", CONFIG,
            "--resume", stdout=log, stderr=subprocess.STDOUT)
    return split, output


def main():
    os.chdir(ROOT)
    os.environ.update(LITTLE_TTS_FRONTEND="own", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # CMUdict-based pronunciation tables contain lexical data, not generated audio
    # or TTS weights. Preserve their own license notices alongside the human run.
    dictionary = BASE / "phonemizer"
    shutil.copytree(ROOT / "phonemizer", dictionary, dirs_exist_ok=True)
    os.environ["DIZIONE_DATA"] = str(dictionary)
    from little_tts.config import Config
    from little_tts.journal import file_hash
    from huggingface_hub import HfApi, snapshot_download
    config = Config(speakers=("92", "6097"))
    write_json(CONFIG, config.to_dict())
    shutil.copyfile(ROOT / "DATA_LICENSE.txt", BASE / "DATA_LICENSE.txt")
    rows = dataset_rows(config)
    write_json(BASE / "provenance.json", dict(dataset=DATASET, revision=REVISION, subset="clean",
               recording_type="real human", original_sample_rate=44100, sample_rate=24000,
               license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
               initialize_from=None, splits={s:len(v) for s,v in rows.items()}))
    print("Real-human dataset:", json.dumps({s:len(v) for s,v in rows.items()}), flush=True)
    tasks = []
    manifests = BASE / "manifests"
    manifests.mkdir(exist_ok=True)
    for split, records in rows.items():
        for index in range(0, len(records), 128):
            manifest = manifests / f"hifi_{split}_{index // 128:04d}.jsonl"
            manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records[index:index+128]))
            tasks.append((split, manifest))
    prepared = []
    workers = int(os.environ.get("PREPARE_WORKERS", "20"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(prepare, tasks), 1):
            prepared.append(result)
            print("Prepared human shard", index, "/", len(tasks), flush=True)
    FEATURES.mkdir(exist_ok=True)
    write_json(FEATURES / "features.json", dict(fingerprint=config.acoustic_fingerprint, fields=config.acoustic_config))
    for split in rows:
        merged, rejected, shard_metadata = [], [], []
        (FEATURES / split).mkdir(exist_ok=True)
        for shard_index, (kind, output) in enumerate(prepared):
            if kind != split:
                continue
            shard_metadata.append(dict(path=str(output), sha256=file_hash(output / (split + ".preparation.json"))))
            for suffix, destination in ((".jsonl", merged), (".rejected.jsonl", rejected)):
                for line in (output / (split + suffix)).read_text().splitlines():
                    entry = json.loads(line)
                    entry["record_key"] = f"{shard_index}:{entry['record_key']}"
                    if suffix == ".jsonl":
                        target = FEATURES / entry["path"]
                        if not target.exists():
                            os.link(output / entry["path"], target)
                    destination.append(entry)
            
        for suffix, records in ((".jsonl", merged), (".rejected.jsonl", rejected)):
            (FEATURES / (split + suffix)).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records))
        write_json(FEATURES / (split + ".preparation.json"), dict(complete=True, accepted=len(merged),
                   rejected=len(rejected), identity=dict(acoustic=config.acoustic_fingerprint,
                   method="merged parallel little_tts.prepare journals", shards=shard_metadata)))
        print("Human features:", split, "accepted", len(merged), "rejected", len(rejected), flush=True)
        if len(merged) < .95 * len(rows[split]):
            raise RuntimeError(f"Unexpectedly high feature rejection rate for {split}")
    # Populate the cache's main reference, verifying it resolves to the pinned aligner.
    model = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    if HfApi().model_info(model).sha != ALIGNER_REVISION:
        raise RuntimeError("Aligner revision changed; review before proceeding")
    snapshot_download(model)
    run(sys.executable, "-m", "little_tts.annotate", "--features", FEATURES,
        "--output", ANNOTATIONS, "--config", CONFIG, "--device", "cuda",
        "--dataset-revision", f"{DATASET}@{REVISION}", "--resume")
    summary = json.loads((ANNOTATIONS / "summary.json").read_text())
    for split in rows:
        stats = summary["splits"][split]
        if stats["accepted"] < .5 * len(rows[split]):
            raise RuntimeError(f"Review alignment quality before training: {split}: {stats}")
    output = ROOT / "runs/ddsp_hifitts_human"
    # The trainer requires an empty directory on a new run; copy notices afterward.
    args = [sys.executable, "-m", "little_tts.train", "--train", ANNOTATIONS / "train.jsonl",
            "--validation", ANNOTATIONS / "validation.jsonl", "--config", CONFIG,
            "--output", output, "--steps", "104000", "--gan-start", "8000",
            "--gan-ratio", "0.7", "--teacher-hold", "3000", "--teacher-ramp", "7000",
            "--teacher-final", "0", "--max-seconds", "86400", "--save-every", "1000",
            "--stop-file", ROOT / "STOP_HUMAN_TRAINING"]
    if (output / "last.pt").exists():
        args += ["--resume", output / "last.pt"]
    print("Starting real-human training from scratch:", " ".join(map(str,args)), flush=True)
    train_with_previews(args, output)
    shutil.copyfile(BASE / "DATA_LICENSE.txt", output / "DATA_LICENSE.txt")
    shutil.copyfile(BASE / "provenance.json", output / "data_provenance.json")


if __name__ == "__main__":
    main()
