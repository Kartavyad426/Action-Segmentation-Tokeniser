import torch
import numpy as np
import os

import pytest

from tokenizer.train import load_checkpoint, train_tokenizer
from tokenizer.windowing import split_available_demos

requires_data = pytest.mark.skipif(
    not os.path.exists("data/reassemble"), reason="REASSEMBLE demos not downloaded"
)


@requires_data
def test_train_tokenizer_runs_and_saves_a_loadable_checkpoint(tmp_path):
    train_paths, val_paths = split_available_demos()
    demo_paths = (train_paths + val_paths)[:3]
    assert len(demo_paths) >= 1, "expected at least one downloaded demo for this test"

    out_path = str(tmp_path / "tokenizer.pt")
    result = train_tokenizer(demo_paths, out_path, epochs=2, batch_size=8, device="cpu")

    assert "final_loss" in result
    assert os.path.exists(out_path)

    model, mean, std, config = load_checkpoint(out_path)
    assert mean.shape[1] == std.shape[1]
    assert config["window"] == 15


def _tiny_demo_telemetry(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float32)


def test_history_records_one_training_loss_per_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokenizer.train._load_all_telemetry", lambda paths: [_tiny_demo_telemetry(tmp_path, seed=i) for i, _ in enumerate(paths)]
    )

    result = train_tokenizer(["a", "b"], str(tmp_path / "t.pt"), epochs=4, window=10, batch_size=8, device="cpu")

    assert len(result["history"]) == 4
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3, 4]
    assert all(isinstance(h["train_loss"], float) for h in result["history"])


def test_training_loss_is_an_epoch_average_not_the_last_batch(tmp_path, monkeypatch):
    # final_loss was the last training batch's loss -- a single noisy sample, not a summary.
    monkeypatch.setattr(
        "tokenizer.train._load_all_telemetry", lambda paths: [_tiny_demo_telemetry(tmp_path, seed=i) for i, _ in enumerate(paths)]
    )

    result = train_tokenizer(["a", "b"], str(tmp_path / "t.pt"), epochs=2, window=10, batch_size=8, device="cpu")

    assert result["history"][-1]["train_loss"] != result["final_loss"]


def test_validation_loss_is_recorded_when_val_demos_are_given(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokenizer.train._load_all_telemetry", lambda paths: [_tiny_demo_telemetry(tmp_path, seed=hash(p) % 100) for p in paths]
    )

    result = train_tokenizer(
        ["a", "b"], str(tmp_path / "t.pt"), val_paths=["c"], epochs=3, window=10, batch_size=8, device="cpu"
    )

    assert all(isinstance(h["val_loss"], float) for h in result["history"])
    assert len(result["history"]) == 3


def test_validation_loss_is_absent_when_no_val_demos_are_given(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tokenizer.train._load_all_telemetry", lambda paths: [_tiny_demo_telemetry(tmp_path, seed=i) for i, _ in enumerate(paths)]
    )

    result = train_tokenizer(["a"], str(tmp_path / "t.pt"), epochs=2, window=10, batch_size=8, device="cpu")

    assert all(h["val_loss"] is None for h in result["history"])


def test_validation_uses_the_training_normalization_stats(tmp_path, monkeypatch):
    # Recomputing mean/std over the validation demos would leak their distribution into the
    # evaluation and make the two losses incomparable.
    monkeypatch.setattr(
        "tokenizer.train._load_all_telemetry", lambda paths: [_tiny_demo_telemetry(tmp_path, seed=hash(p) % 100) for p in paths]
    )

    without_val = train_tokenizer(["a", "b"], str(tmp_path / "x.pt"), epochs=1, window=10, batch_size=8, device="cpu")
    with_val = train_tokenizer(
        ["a", "b"], str(tmp_path / "y.pt"), val_paths=["c"], epochs=1, window=10, batch_size=8, device="cpu"
    )

    x = torch.load(str(tmp_path / "x.pt"), weights_only=False)
    y = torch.load(str(tmp_path / "y.pt"), weights_only=False)
    assert np.allclose(x["mean"], y["mean"])
    assert np.allclose(x["std"], y["std"])
