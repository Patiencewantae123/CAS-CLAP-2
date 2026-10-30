# =============================================================================
# Build image + ground-truth audio pairs from an AudioCaps split (test or val).
#
# AudioCaps ships captions and YouTube ids, not media. For each sampled clip
# this downloads only the 10 s segment, then extracts
#   audio/<ytid>.wav  : the real 16 kHz mono soundtrack (ground truth)
#   frames/<ytid>.jpg : the video frame at the middle of the segment (input)
# Videos that are no longer available are skipped and listed in failed.json.
#
# Usage:
#   py -m scripts.download_data --split test --n 100
#   py -m scripts.download_data --split val --n 100
# =============================================================================

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg

from casv.log import setup_logging

log = logging.getLogger("casv")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "audiocaps"
CLIP_SECONDS = 10
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_clips(split):
    """AudioCaps clips of a split: start time and all human captions per YouTube id."""
    clips = defaultdict(lambda: {"captions": []})
    with open(DATA_DIR / "splits" / f"{split}.csv", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            clip = clips[row["youtube_id"]]
            clip["start"] = int(row["start_time"])
            clip["captions"].append(row["caption"].strip())
    return clips


def fetch_clip(ytid, start, out_dir):
    """Download the 10 s segment; save 16 kHz audio and the middle frame. Returns (ok, error)."""
    wav, jpg = out_dir / "audio" / f"{ytid}.wav", out_dir / "frames" / f"{ytid}.jpg"
    if wav.exists() and jpg.exists():
        return True, ""
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        download = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "--js-runtimes",
                "node",
                "-f",
                "18/best[height<=480][acodec!=none]/best",
                "--download-sections",
                f"*{start}-{start + CLIP_SECONDS}",
                "--ffmpeg-location",
                FFMPEG,
                "-o",
                str(video),
                f"https://www.youtube.com/watch?v={ytid}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if download.returncode != 0 or not video.exists():
            return False, (download.stderr.strip().splitlines() or ["download failed"])[-1]
        audio = subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            capture_output=True,
            text=True,
        )
        frame = subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(CLIP_SECONDS / 2),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(jpg),
            ],
            capture_output=True,
            text=True,
        )
        if audio.returncode != 0 or frame.returncode != 0 or not jpg.exists():
            wav.unlink(missing_ok=True)
            return False, (audio.stderr or frame.stderr).strip()[-200:]
    return True, ""


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch AudioCaps clips as image + audio pairs.")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--n", type=int, default=100, help="Number of pairs to collect.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="Default: data/audiocaps/pairs/<split>")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else DATA_DIR / "pairs" / args.split
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    clips = load_clips(args.split)
    order = sorted(clips)
    random.Random(args.seed).shuffle(order)
    log.info(f"{len(clips)} AudioCaps {args.split} clips; collecting {args.n} (seed {args.seed})")

    manifest, failed = [], {}
    pending = iter(order)
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {}

        def submit_next():
            ytid = next(pending, None)
            if ytid is not None:
                futures[pool.submit(fetch_clip, ytid, clips[ytid]["start"], out_dir)] = ytid

        for _ in range(args.workers * 2):
            submit_next()
        while futures and len(manifest) < args.n:
            future = next(as_completed(futures))
            ytid = futures.pop(future)
            ok, error = future.result()
            if ok:
                manifest.append(
                    {
                        "id": ytid,
                        "image": f"frames/{ytid}.jpg",
                        "gt_audio": f"audio/{ytid}.wav",
                        "captions": clips[ytid]["captions"],
                        "start_time": clips[ytid]["start"],
                    }
                )
                log.info(f"    [{len(manifest)}/{args.n}] {ytid}")
            else:
                failed[ytid] = error
            if len(manifest) + len(futures) < args.n:
                submit_next()
        for future in futures:
            future.cancel()

    manifest.sort(key=lambda m: order.index(m["id"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest[: args.n], indent=2))
    (out_dir / "failed.json").write_text(json.dumps(failed, indent=2))
    log.info(f"{len(manifest[: args.n])} pairs in {out_dir}; {len(failed)} unavailable videos skipped")


if __name__ == "__main__":
    main()
