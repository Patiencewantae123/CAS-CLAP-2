# =============================================================================
# CAS-V vocabularies built from public datasets
#
#   scenes  : Places365 categories (+ official indoor/outdoor annotation)
#   objects : ImageNet-1k class names
#   sounds  : AudioCaps *train* captions (test captions are never used here,
#             so evaluation on the AudioCaps test split has no leakage)
#
# SOUND_SOURCES lists sound-producing things that are visible when present.
# The verifier flags a candidate caption that names one of these while the
# image shows no evidence of it (a hallucinated sound source).
# =============================================================================

import csv
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# word pattern in caption -> what CLIP should see in the image
SOUND_SOURCES = {
    r"\bdogs?\b|\bpupp(y|ies)\b|\bbark": "a dog",
    r"\bcats?\b|\bmeow|\bpurr": "a cat",
    r"\bbirds?\b|\bchirp|\btweet|\bpigeons?\b|\bcrows?\b": "birds",
    r"\bchickens?\b|\brooster|\bhens?\b|\bcluck": "chickens",
    r"\bducks?\b|\bquack|\bgeese\b|\bgoose\b": "ducks or geese",
    r"\bgoats?\b|\bsheep\b|\bbleat": "goats or sheep",
    r"\bcows?\b|\bcattle\b|\bmoo": "cows",
    r"\bhorses?\b|\bneigh|\bgallop|\bhooves\b": "a horse",
    r"\bpigs?\b|\boink": "pigs",
    r"\bfrogs?\b|\bcroak": "a frog",
    r"\bbees?\b|\binsects?\b|\bbuzz": "insects",
    r"\bbab(y|ies)\b|\binfant": "a baby",
    r"\bcrowd\b|\baudience\b|\bapplau|\bcheer": "a crowd of people",
    r"\bcars?\b|\btraffic\b|\bvehicles?\b|\bhonk": "cars on a road",
    r"\btrucks?\b|\bbus(es)?\b": "a truck or bus",
    r"\bmotorcycles?\b|\bmotorbikes?\b|\bdirt bike": "a motorcycle",
    r"\btrains?\b|\brailroad\b|\blocomotive": "a train",
    r"\bairplanes?\b|\baircraft\b|\bplanes?\b|\bjets?\b": "an airplane",
    r"\bhelicopters?\b": "a helicopter",
    r"\bboats?\b|\bmotorboat|\bship\b": "a boat",
    r"\bsirens?\b|\bambulance|\bpolice car|\bfire truck": "an emergency vehicle",
    r"\bengines?\b|\bidling\b|\brevs?\b|\brevving\b|\bmotor\b": "a motor vehicle with an engine",
    r"\bwaves?\b|\bocean\b|\bsurf\b|\bseashore\b": "the sea",
    r"\brain(s|ing|fall)?\b": "rain",
    r"\bstream\b|\bwaterfall\b|\briver\b|\bbrook\b": "a stream or river",
    r"\bfire\b|\bcrackl": "fire",
    r"\bfireworks?\b": "fireworks",
    r"\bguns?(fire|shots?)?\b|\bgunshots?\b|\brifle": "a gun",
    r"\bguitar": "a guitar",
    r"\bpiano\b": "a piano",
    r"\bdrums?\b": "drums",
    r"\btyping\b|\bkeyboard\b": "a computer keyboard",
    r"\bsewing machine": "a sewing machine",
    r"\bvacuum\b": "a vacuum cleaner",
    r"\bdrill(ing)?\b|\bpower tool|\bsaw(ing)?\b|\bsander\b": "a power tool",
    r"\bhammer(ing|s)?\b": "a hammer",
    r"\btoilet\b|\bflush": "a toilet",
    r"\bdishes\b|\bplates?\b|\bsilverware\b|\butensils?\b": "dishes",
    r"\bfr(y|ying|ies)\b|\bsizzl": "food frying in a pan",
    r"\bclock\b|\btick(ing)?\b": "a clock",
    r"\bbells?\b|\bchimes?\b": "a bell",
}

# caption keywords that pin a sound to an environment
INDOOR_WORDS = r"\btyping\b|\bkeyboard\b|\bvacuum\b|\btoilet\b|\bsewing machine\b|\bdishes\b|\bindoors?\b|\btelevision\b|\bmicrowave\b|\bkitchen\b"
OUTDOOR_WORDS = r"\btraffic\b|\bwind\b|\bthunder\b|\bwaves?\b|\bocean\b|\bbirds?\b|\boutdoors?\b|\bstreet\b|\bstream\b|\bwaterfall\b|\brain\b|\bhelicopter\b|\bairplane\b|\btrain\b"


def _clean_places_name(raw):
    # "/f/field/wild" -> "wild field", "/b/bedroom" -> "bedroom"
    parts = raw.strip("/").split("/")[1:]
    name = " ".join(reversed(parts)) if len(parts) > 1 else parts[0]
    return name.replace("_", " ")


def load_places365():
    names, envs = [], []
    io_map = {}
    for line in (DATA_DIR / "vocab" / "IO_places365.txt").read_text().split("\n"):
        if line.strip():
            raw, flag = line.split()
            io_map[raw] = "indoor" if flag == "1" else "outdoor"
    for line in (DATA_DIR / "vocab" / "categories_places365.txt").read_text().split("\n"):
        if line.strip():
            raw = line.split()[0]
            names.append(_clean_places_name(raw))
            envs.append(io_map[raw])
    return names, envs


def load_imagenet():
    return json.loads((DATA_DIR / "vocab" / "imagenet_simple_labels.json").read_text())


def load_audiocaps_captions(split="train"):
    with open(DATA_DIR / "audiocaps" / f"{split}.csv", newline="", encoding="utf-8") as file:
        captions = {row["caption"].strip().rstrip(".") for row in csv.DictReader(file)}
    return sorted(c for c in captions if c)


def mentioned_sources(caption):
    text = caption.lower()
    return [visual for pattern, visual in SOUND_SOURCES.items() if re.search(pattern, text)]


def caption_environment(caption):
    text = caption.lower()
    indoor = bool(re.search(INDOOR_WORDS, text))
    outdoor = bool(re.search(OUTDOOR_WORDS, text))
    if indoor and not outdoor:
        return "indoor"
    if outdoor and not indoor:
        return "outdoor"
    return "any"
