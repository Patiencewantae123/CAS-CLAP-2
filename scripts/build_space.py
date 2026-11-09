# =============================================================================
# Bundle the CAS-V board for a Hugging Face Space (Gradio SDK on ZeroGPU).
#
# Copies only what the Space needs into dist/hf_space/ (git-ignored):
#   app_gradio.py, board.py, casv/*.py, scripts/figures.py
#   data/vocab/, AudioCaps train captions, tuned verifier config (ablation),
#   the paper's evaluation graphs (shown under "More info")
#   the cached CLIP text embeddings the verifier loads (else the Space would
#   spend minutes re-encoding ~46k captions on every start)
#   space/README.md (Space settings) and space/requirements.txt
# Nothing else is copied, so git-ignored files such as the listening-test key
# or downloaded AudioCaps media can never reach the Space.
#
# Usage:
#   py -m scripts.build_space                      # bundle only
#   py -m scripts.deploy_space <user>/<space>      # bundle, log in, upload, follow the build
# =============================================================================

import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from casv.log import setup_logging
from casv.verifier import CLIP_MODEL_ID, CASVerifier, text_cache_path

log = logging.getLogger("casv")
ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/Patiencewantae123/CAS-CLAP-2"
FILES = [
    "app_gradio.py",
    "board.py",
    "scripts/figures.py",
    "data/audiocaps/splits/train.csv",
    "results/tuning_test400/casv_tuned_config.json",
    "journal_results/graphs/fig1_scene_recognition.png",
    "journal_results/graphs/fig2_prompt_selection.png",
    "journal_results/graphs/fig3_generated_audio_clap.png",
    "journal_results/graphs/fig4_verification_statistics.png",
    "LICENSE",
]
FOLDERS = {"casv": "*.py", "data/vocab": "*"}
SPACE_FILES = {"space/README.md": "README.md", "space/requirements.txt": "requirements.txt"}
MARKER = "BUILD_INFO.txt"  # provenance on the Space; only folders carrying it are deleted by a rebuild
DEFAULT_OUT = ROOT / "dist" / "hf_space"


class CacheRecorder:
    """Stands in for CLIPRecognizer: records which cached text embeddings the verifier loads."""

    model_id = CLIP_MODEL_ID

    def __init__(self):
        self.paths = []

    def encode_texts(self, texts):
        path = text_cache_path(texts, self.model_id)
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}: start app.py or app_gradio.py once to create it.")
        self.paths.append(path)
        return np.load(path)


def build_info() -> str:
    """Which commit the bundle was built from, and whether the working tree had changes."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "HEAD") or "unknown (not a git checkout)"
    dirty = " plus uncommitted changes" if git("status", "--porcelain") else ""
    return (
        f"Built by scripts/build_space.py on {datetime.now():%Y-%m-%d %H:%M} from {REPO_URL}\n"
        f"commit {commit}{dirty}. Regenerate it there rather than editing these files.\n"
    )


def copy(source: Path, out: Path) -> None:
    target = out / source.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build(out: Path = DEFAULT_OUT) -> list[Path]:
    """Write the Space bundle to out (replacing an earlier build); returns its files."""
    out = out.resolve()
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this script; choose another --out.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text(build_info(), encoding="utf-8")

    for name in FILES:
        copy(ROOT / name, out)
    for folder, pattern in FOLDERS.items():
        for source in sorted((ROOT / folder).glob(pattern)):
            if source.is_file():
                copy(source, out)
    for source, target in SPACE_FILES.items():
        shutil.copyfile(ROOT / source, out / target)

    recorder = CacheRecorder()
    CASVerifier(recorder)  # loads the vocabularies and asks for exactly the embeddings the Space needs
    for path in recorder.paths:
        copy(path, out)

    files = [p for p in out.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 2**20
    log.info(f"Space bundle: {len(files)} files, {size_mb:.0f} MB in {out}")
    return files


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Bundle the CAS-V board for a Hugging Face Space.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    out = Path(parser.parse_args().out).resolve()
    for path in build(out):
        log.info(f"  {path.relative_to(out).as_posix()}")
    log.info("Upload it with: py -m scripts.deploy_space <user>/<space>")


if __name__ == "__main__":
    main()
