"""Assemble the static demo's local assets from this release (no downloads)."""
from pathlib import Path
import shutil

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
(WEB / "assets").mkdir(exist_ok=True)
shutil.copyfile(ROOT / "onnx/model.onnx", WEB / "assets/model.onnx")
shutil.copyfile(ROOT / "onnx/model.json", WEB / "assets/config.json")
shutil.copytree(ROOT / "phonemizer", WEB / "assets/phonemizer", dirs_exist_ok=True)
print("Static assets ready. Serve web/ over HTTP, for example: python -m http.server --directory web 8000")
