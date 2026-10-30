"""The tuner's vectorised selection must pick exactly what CASVerifier.decide() picks."""

import random

import numpy as np
import pytest

from casv import vocab
from casv.verifier import CASVerifier, VerifierConfig
from scripts.tune_casv import candidate_arrays, configs, select

CAPTIONS = [
    "A dog barks while traffic passes",
    "Typing on a computer keyboard",
    "Rain falls on a roof as thunder rumbles",
    "A man speaks",
    "Birds chirp as wind blows",
    "A toilet flushes",
    "An engine idles then revs",
    "Ocean waves crash on the shore",
    "A cat meows and purrs",
    "Dishes clatter in a kitchen",
    "People talk in the background",
    "A helicopter flies overhead",
]
SOURCES = list(dict.fromkeys(vocab.SOUND_SOURCES.values()))


def random_record(rng: random.Random, k: int = 20) -> dict:
    p_indoor = rng.random()
    return {
        "candidates": [
            {
                "caption": rng.choice(CAPTIONS),
                "visual": rng.random(),
                "scene_support": rng.random(),
                "object_support": rng.random(),
            }
            for _ in range(k)
        ],
        "env_probs": {"indoor": p_indoor, "outdoor": 1.0 - p_indoor},
        "source_z": {s: rng.gauss(0, 1) for s in SOURCES},
        "top_scenes": [("street", rng.random())],
        "top_objects": [("car", rng.random())],
    }


@pytest.fixture(scope="module")
def verifier() -> CASVerifier:
    instance = CASVerifier.__new__(CASVerifier)  # decide() needs no models
    instance.config = VerifierConfig()
    return instance


def test_vectorised_selection_matches_decide(verifier):
    rng = random.Random(0)
    all_configs = list(configs())
    sampled = rng.sample(all_configs, 200) + [VerifierConfig()]
    for _ in range(100):
        rec = random_record(rng)
        arrays = candidate_arrays(rec)
        for cfg in rng.sample(sampled, 20):
            index = select(arrays, cfg)
            expected = rec["candidates"][index]["caption"] if index >= 0 else arrays["fallback"]
            assert verifier.decide(rec, cfg)["casv_prompt"] == expected


def test_baseline_is_top_retrieved_candidate(verifier):
    rec = random_record(random.Random(1))
    assert verifier.decide(rec)["baseline_prompt"] == rec["candidates"][0]["caption"]


def test_conflicting_candidate_is_rejected(verifier):
    rec = random_record(random.Random(2), k=2)
    rec["candidates"][0]["caption"] = "A dog barks"
    rec["source_z"]["a dog"] = -3.0  # dog clearly not visible
    verdict = verifier.decide(rec)["verdicts"][0]
    assert verdict["decision"] == "REJECT"
    assert any("a dog" in c for c in verdict["conflicts"])


def test_search_space_size():
    assert len(list(configs())) == 66 * 4 * 5 * 5
    assert all(np.isclose(c.w_visual + c.w_scene + c.w_object, 1.0) for c in configs())
