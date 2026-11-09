"""CLAP scoring must not depend on a random crop."""

from types import SimpleNamespace

import numpy as np
import torch

from casv.audio import CLAPScorer


class FakeProcessor:
    feature_extractor = SimpleNamespace(nb_max_samples=480000)

    def __init__(self):
        self.seen = []

    def __call__(self, audio, sampling_rate, return_tensors):
        self.seen.append(audio)
        return SimpleNamespace(to=lambda device: {})


def fake_scorer():
    scorer = CLAPScorer.__new__(CLAPScorer)
    scorer.device = "cpu"
    scorer.processor = FakeProcessor()
    scorer.model = SimpleNamespace(get_audio_features=lambda: torch.ones((1, 4)))
    return scorer


def test_clip_over_ten_seconds_keeps_its_first_ten_seconds():
    scorer = fake_scorer()
    audio = np.arange(480384, dtype=np.float32)
    scorer.audio_embedding(audio, CLAPScorer.sample_rate)
    assert np.array_equal(scorer.processor.seen[0], audio[:480000])


def test_short_clip_is_not_cropped():
    scorer = fake_scorer()
    audio = np.arange(240000, dtype=np.float32)
    scorer.audio_embedding(audio, CLAPScorer.sample_rate)
    assert np.array_equal(scorer.processor.seen[0], audio)
