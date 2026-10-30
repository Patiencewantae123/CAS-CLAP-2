"""Short concept phrases and sound types derived from a verified caption (used in figures and the app)."""

import re

CLAUSE_SPLIT = r",|;|\band then\b|\bthen\b|\bfollowed by\b|\bwhile\b|\bas\b|\band\b|\balong with\b|\bwith\b"

LEADING = r"^(a|an|the|some|several|multiple|series of|a series of|sounds? of|noises? of)\s+"
TRAILING_AUX = r"\s+(is|are|was|were|being|been|occurs|occur)(\s+.*)?$"
STOP_END = {"on", "of", "in", "by", "to", "its", "the", "a", "an", "is", "are", "at", "into", "from"}
NOT_A_SOURCE = {
    "one",
    "another",
    "it",
    "they",
    "this",
    "that",
    "low",
    "muffled",
    "continuous",
    "loud",
    "soft",
    "briefly",
    "repeatedly",
    "background",
    "distance",
}


def clean_phrase(phrase):
    """Lower-case a caption clause and drop leading quantifiers and trailing auxiliaries."""
    phrase = phrase.strip(" .").lower()
    for _ in range(2):
        phrase = re.sub(LEADING, "", phrase)
    return re.sub(TRAILING_AUX, "", phrase).strip()


def verified_concepts(caption, limit=4):
    """Short concept phrases taken from the verified caption."""
    parts = [clean_phrase(p) for p in re.split(CLAUSE_SPLIT, caption, flags=re.IGNORECASE)]
    parts = [p for p in parts if p and p.split()[0] not in NOT_A_SOURCE]
    return list(dict.fromkeys(parts))[:limit]


def short_head(concept, max_words=3, keep_whole=5):
    """Short form of a concept for the sound-type box."""
    if len(concept.split()) <= keep_whole:
        return concept
    words = concept.split()[:max_words]
    while words and words[-1] in STOP_END:
        words.pop()
    return " ".join(words)


def sound_type(concepts):
    """Sound type shown in the figure: up to two concept heads."""
    heads = []
    for concept in concepts:
        head = short_head(concept)
        if head and head not in heads:
            heads.append(head)
        if len(heads) == 2:
            break
    return " / ".join(heads).capitalize()
