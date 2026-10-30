# =============================================================================
# One AudioCaps *test* clip per category for the qualitative figure
# (Busy Road, Office, Beach, Forest, Cloth Scratch, Leaves, Metal Hit).
#
# Selection rule, fixed before any result is seen: shuffle test clips with
# --seed, take the first clip where at least --min-captions of its human
# captions match the category pattern and the video is still available.
# The rule and the matching captions are written to the manifest.
#
# Usage:
#   py -m scripts.select_examples
# =============================================================================

import argparse
import json
import logging
import random
import re
from pathlib import Path

from casv.log import setup_logging
from scripts.download_data import DATA_DIR, fetch_clip, load_clips

log = logging.getLogger("casv")

CATEGORIES = {
    "busy road": r"\btraffic\b",
    "office": r"\btyping\b|\bkeyboard\b",
    "beach": r"\bwaves?\b.*\b(crash|break|beach|shore)|\b(beach|shore)\b",
    "forest": r"\bbirds?\b.*\b(chirp|tweet|sing)\w*.*\bwind\b|\bwind\b.*\bbirds?\b|\bforest\b",
    "cloth scratch": r"\b(cloth|fabric|clothes|clothing)\b|\bscratch\w*",
    "leaves": r"\bleaves\b|\bfoliage\b",
    # impact wording must follow "metal" directly (a clank somewhere in a gunshot clip is not a metal hit)
    "metal hit": r"\bhammer\w*|\bmetal\w*\s+(\w+\s+)?(clank|clack|clang|bang|hit|strik|tap|ping)\w*",
}


def candidates(label, clips, order, min_captions):
    """Clips in seeded order whose captions match the category, with the matching captions."""
    for ytid in order:
        matched = [c for c in clips[ytid]["captions"] if re.search(CATEGORIES[label], c.lower())]
        if len(matched) >= min_captions:
            yield ytid, matched


def contact_sheet(label, ytids, out_dir):
    """Save the candidate frames of one category side by side for visual screening."""
    from PIL import Image, ImageDraw

    width, height = 320, 240
    sheet = Image.new("RGB", (width * len(ytids), height + 22), "white")
    for i, ytid in enumerate(ytids):
        sheet.paste(
            Image.open(out_dir / "frames" / f"{ytid}.jpg").convert("RGB").resize((width, height)), (i * width, 22)
        )
        ImageDraw.Draw(sheet).text((i * width + 6, 5), f"{i + 1}. {ytid}", fill="black")
    path = out_dir / "screening" / f"{label.replace(' ', '_')}.png"
    path.parent.mkdir(exist_ok=True)
    sheet.save(path)
    return path


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch one AudioCaps test clip per figure category.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-captions", type=int, default=2)
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Fetch the first N available clips per category and write frame contact "
        "sheets for visual screening; no manifest is written.",
    )
    parser.add_argument("--only", default=None, help="Comma-separated categories to preview.")
    parser.add_argument(
        "--screening", default=None, help='JSON {"rejected": {ytid: reason}} from visual screening of frames only.'
    )
    args = parser.parse_args()

    out_dir = DATA_DIR / "pairs" / "categories"
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    clips = load_clips("test")
    order = sorted(clips)
    random.Random(args.seed).shuffle(order)
    screening = json.loads(Path(args.screening).read_text(encoding="utf-8")) if args.screening else {}
    rejected, relabel = screening.get("rejected", {}), screening.get("relabel", {})

    manifest, skipped = [], {}
    for label in CATEGORIES:
        if args.only and label not in args.only.split(","):
            continue
        fetched = []
        for ytid, matched in candidates(label, clips, order, args.min_captions):
            if ytid in rejected or any(m["id"] == ytid for m in manifest):
                continue
            ok, error = fetch_clip(ytid, clips[ytid]["start"], out_dir)
            if not ok:
                skipped[ytid] = error
                continue
            fetched.append(ytid)
            if args.preview and len(fetched) < args.preview:
                continue
            if not args.preview:
                manifest.append(
                    {
                        "id": ytid,
                        "label": relabel.get(label, label),
                        "category": label,
                        "image": f"frames/{ytid}.jpg",
                        "gt_audio": f"audio/{ytid}.wav",
                        "captions": clips[ytid]["captions"],
                        "start_time": clips[ytid]["start"],
                        "matched_captions": matched,
                    }
                )
                log.info(f"    {label:14} {ytid}  {matched[0]}")
            break
        if args.preview:
            log.info(f"    {label:14} {contact_sheet(label, fetched, out_dir)}")
        elif not fetched:
            log.info(f"    {label:14} no available clip")

    if args.preview:
        return
    rule = {
        "split": "test",
        "seed": args.seed,
        "min_matching_captions": args.min_captions,
        "patterns": CATEGORIES,
        "visual_screening": "frames only, by an author, before any audio was generated; "
        "the first non-rejected clip in seeded order is used",
        "rejected_on_screening": rejected,
        "relabelled": relabel,
        "unavailable": skipped,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "selection_rule.json").write_text(json.dumps(rule, indent=2))
    log.info(f"{len(manifest)}/{len(CATEGORIES)} categories in {out_dir}")


if __name__ == "__main__":
    main()
