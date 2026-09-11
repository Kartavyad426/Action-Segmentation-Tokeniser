"""Runs must be reproducible, and repeats must differ only by seed."""

import numpy as np
import pytest

from tokenizer.train import tokenizer_fingerprint


def test_tokenizer_seed_changes_the_fingerprint():
    # A different seed is a different tokenizer, so a checkpoint from one must never be
    # silently reused for the other.
    a = tokenizer_fingerprint(["a.h5"], window=15, seed=0)
    b = tokenizer_fingerprint(["a.h5"], window=15, seed=1)

    assert a != b


def test_same_seed_gives_the_same_fingerprint():
    a = tokenizer_fingerprint(["a.h5"], window=15, seed=3)
    b = tokenizer_fingerprint(["a.h5"], window=15, seed=3)

    assert a == b


@pytest.fixture
def fake_telemetry(monkeypatch):
    def _load(paths):
        rng = np.random.default_rng(0)
        return [rng.standard_normal((300, 3)).astype(np.float32) for _ in paths]

    monkeypatch.setattr("tokenizer.train._load_all_telemetry", _load)


def test_the_same_tokenizer_seed_reproduces_the_same_losses(tmp_path, fake_telemetry):
    from tokenizer.train import train_tokenizer

    kwargs = dict(epochs=2, window=10, num_codes=8, batch_size=16, device="cpu", verbose=False)
    a = train_tokenizer(["x", "y"], str(tmp_path / "a.pt"), seed=7, **kwargs)
    b = train_tokenizer(["x", "y"], str(tmp_path / "b.pt"), seed=7, **kwargs)

    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]


def test_a_different_tokenizer_seed_gives_a_different_trajectory(tmp_path, fake_telemetry):
    from tokenizer.train import train_tokenizer

    kwargs = dict(epochs=2, window=10, num_codes=8, batch_size=16, device="cpu", verbose=False)
    a = train_tokenizer(["x", "y"], str(tmp_path / "a.pt"), seed=1, **kwargs)
    b = train_tokenizer(["x", "y"], str(tmp_path / "b.pt"), seed=2, **kwargs)

    assert [h["train_loss"] for h in a["history"]] != [h["train_loss"] for h in b["history"]]
