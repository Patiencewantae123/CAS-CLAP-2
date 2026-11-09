# =============================================================================
# Choose clips for the selected comparison pages: clear examples of named sound
# categories from a finished run. No CLAP score or listening result is used.
#
# A clip is shown for a category when
#   1. at least --min-captions of its human captions match the category pattern,
#   2. on the input frame alone, CLIP ranks that category first among all
#      categories, and the frame is not near black,
#   3. CAS-V changed the prompt, and its verified caption matches the pattern.
#
# Rule 3 depends on what CAS-V produced, so these pages show what the method
# does when it works. They are illustrations, not results: the disclosure saved
# here goes into every page's caption file, and the random-sample pages and
# Table 3 show the typical case.
#
# Usage:
#   py -m scripts.select_figure_clips --run results/audiocaps_test400
# =============================================================================

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

from casv.log import setup_logging
from casv.verifier import CLIPRecognizer, load_image

log = logging.getLogger("casv")

MIN_BRIGHTNESS = 30  # mean grey level of the frame, 0-255

CATEGORIES = {  # label: (pattern for captions, what the frame should show)
    "busy road": (r"\btraffic\b|\b(cars?|vehicles?) (pass|driv|go)\w*", "a busy road with cars"),
    "office": (r"\btyping\b|\bkeyboard\b", "hands typing on a computer keyboard"),
    "beach": (r"\bwaves?\b|\bocean\b|\bbeach\b|\bshore\b|\bsurf\b", "ocean waves at a beach"),
    "birds": (r"\bbirds?\b", "trees and birds in nature"),
    "leaves": (r"\bleaves\b|\bfoliage\b", "leaves and plants"),
    "metal hit": (
        r"\bhammer\w*|\bmetal\w*\s+(\w+\s+)?(clank|clack|clang|bang|hit|strik|tap|ping)\w*",
        "a person hitting metal with a tool",
    ),
    "cloth": (r"\b(cloth|fabric)\b|\bscratch\w*", "hands rubbing cloth or fabric"),
    "stream": (r"\bstream\b|\bbrook\b|\bcreek\b|\briver\b|\bwater (flow|run|trickl|gurgl)\w*", "a stream or river"),
    "rain": (r"\brain\w*", "rain falling"),
    "dog": (r"\bdogs?\b|\bbark\w*", "a dog"),
    "cat": (r"\bcats?\b|\bmeow\w*", "a cat"),
    "farm animals": (
        r"\bsheep\b|\bgoats?\b|\bcows?\b|\bbaa\w*|\bmoo(s|ing)?\b|\bhorses?\b|\bpigs?\b|\boink\w*",
        "farm animals such as sheep, goats, cows or horses",
    ),
    "ducks / pigeons": (r"\bducks?\b|\bpigeons?\b|\bquack\w*|\bcoo(s|ing)?\b", "ducks or pigeons"),
    "baby": (r"\bbaby\b|\binfant\b", "a baby"),
    "train": (r"\btrains?\b|\blocomotive\b|\brailroad\b", "a train on railway tracks"),
    "aircraft": (r"\b(air)?planes?\b|\baircraft\b|\bjet\b", "an airplane"),
    "helicopter": (r"\bhelicopter\b", "a helicopter"),
    "motorcycle": (r"\bmotorcycle\b|\bmotorbike\b|\bdirt bike\b", "a motorcycle"),
    "boat": (r"\bboat\b|\bmotorboat\b|\bspeedboat\b", "a boat on water"),
    "engine": (r"\bengines?\b|\bidling\b|\brevving\b", "a car, truck or vehicle engine"),
    "kitchen": (r"\bsizzl\w*|\bfry\w*", "food frying in a pan"),
    "sewing machine": (r"\bsewing\b", "a sewing machine"),
    "toilet": (r"\btoilet\b", "a toilet"),
    "bells": (r"\bbells?\b|\bchim(e|es|ing)\b", "a bell or clock"),
    "crowd": (r"\bapplau\w*|\bcheer\w*|\bcrowd\b", "a crowd of people"),
    "power tools": (r"\bdrill\w*|\bsawing\b|\bsanding\b|\bpower tool\b", "a person using a power tool"),
    "insects": (r"\bbees?\b|\binsects?\b|\bbuzz\w*", "bees or insects"),
    "siren": (r"\bsirens?\b", "an emergency vehicle"),
}


def caption_labels(captions, min_captions):
    """Categories matched by at least min_captions of the human captions."""
    return [
        label
        for label, (pattern, _) in CATEGORIES.items()
        if sum(bool(re.search(pattern, caption.lower())) for caption in captions) >= min_captions
    ]


def round_robin(pools):
    """One clip per category in turn, so a page shows different categories."""
    queues = {label: list(ids) for label, ids in pools.items()}
    ordered = []
    while any(queues.values()):
        for label, queue in queues.items():
            if queue:
                ordered.append({"id": queue.pop(0), "label": label})
    return ordered


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Choose clear examples of named categories for comparison pages.")
    parser.add_argument("--run", required=True, help="Run directory produced by test.py")
    parser.add_argument("--min-captions", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run)
    items = json.loads((run_dir / "results.json").read_text())["items"]
    recognizer = CLIPRecognizer()
    labels = list(CATEGORIES)
    visual = recognizer.encode_texts([what for _, what in CATEGORIES.values()])

    pools = {label: [] for label in labels}
    n_captions = n_frame = 0
    for item in items:
        matched = caption_labels(item["gt_captions"], args.min_captions)
        if not matched:
            continue
        n_captions += 1
        image = load_image(next((run_dir / item["id"]).glob("input.*")))
        if np.asarray(image.convert("L")).mean() < MIN_BRIGHTNESS:
            continue
        label = labels[int(np.argmax(visual @ recognizer.encode_image(image)))]
        if label not in matched:
            continue
        n_frame += 1
        prompts = item["prompts"]
        if prompts["casv"] != prompts["baseline"] and re.search(CATEGORIES[label][0], prompts["casv"].lower()):
            pools[label].append(item["id"])

    selected = round_robin(pools)
    disclosure = (
        f"Selected examples, not representative. Of the {len(items)} clips in this run, {n_captions} had at least "
        f"{args.min_captions} human captions naming a category and {n_frame} of those had a frame CLIP ranked first "
        f"for that category; the {len(selected)} shown are those where CAS-V also changed the prompt and its verified "
        "caption names the category. No CLAP score was used. See Table 3 for the result over all clips."
    )
    output = {
        "rule": {
            "min_captions": args.min_captions,
            "min_brightness": MIN_BRIGHTNESS,
            "categories": {label: {"pattern": p, "frame": what} for label, (p, what) in CATEGORIES.items()},
        },
        "counts": {
            "clips": len(items),
            "caption_match": n_captions,
            "caption_and_frame_match": n_frame,
            "selected": len(selected),
            "per_category": {label: len(ids) for label, ids in pools.items()},
        },
        "disclosure": disclosure,
        "items": selected,
    }
    (run_dir / "selected_clips.json").write_text(json.dumps(output, indent=2))
    log.info(disclosure)
    log.info(json.dumps(output["counts"]["per_category"]))


if __name__ == "__main__":
    main()
