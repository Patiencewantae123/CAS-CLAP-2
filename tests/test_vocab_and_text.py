"""Vocabulary parsing, caption heuristics and figure text extraction."""

import pytest

from casv import vocab
from casv.text import sound_type, verified_concepts


@pytest.mark.parametrize(
    ("raw", "clean"),
    [("/a/airfield", "airfield"), ("/f/field/wild", "wild field"), ("/b/bus_station/indoor", "indoor bus station")],
)
def test_places_names(raw, clean):
    assert vocab._clean_places_name(raw) == clean


def test_vocabulary_sizes():
    names, envs = vocab.load_places365()
    assert len(names) == len(envs) == 365
    assert set(envs) == {"indoor", "outdoor"}
    assert len(vocab.load_imagenet()) == 1000


@pytest.mark.parametrize(
    ("caption", "source"),
    [
        ("A dog barks loudly", "a dog"),
        ("Typing on a keyboard", "a computer keyboard"),
        ("An engine idles", "a motor vehicle with an engine"),
    ],
)
def test_mentioned_sources(caption, source):
    assert source in vocab.mentioned_sources(caption)


@pytest.mark.parametrize(
    ("caption", "env"),
    [
        ("Typing on a keyboard", "indoor"),
        ("Traffic passes by", "outdoor"),
        ("A man speaks", "any"),
        ("Wind blows while typing", "any"),
    ],
)
def test_caption_environment(caption, env):
    assert vocab.caption_environment(caption) == env


@pytest.mark.parametrize(
    ("caption", "concepts", "kind"),
    [
        ("A series of small pops is followed by an explosion", ["small pops", "explosion"], "Small pops / explosion"),
        ("Wind noise on ocean", ["wind noise on ocean"], "Wind noise on ocean"),
        ("Typing on a computer keyboard", ["typing on a computer keyboard"], "Typing on a computer keyboard"),
    ],
)
def test_figure_text(caption, concepts, kind):
    assert verified_concepts(caption) == concepts
    assert sound_type(verified_concepts(caption)) == kind
