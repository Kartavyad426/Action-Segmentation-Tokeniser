"""A hang or a kill must never cost the whole tokenizer training run."""

import numpy as np
import pytest
import torch

from tokenizer.train import load_checkpoint, train_tokenizer


@pytest.fixture
def fake_telemetry(monkeypatch):
    def _load(paths):
        rng = np.random.default_rng(0)
        return [rng.standard_normal((200, 3)).astype(np.float32) for _ in paths]

    monkeypatch.setattr("tokenizer.train._load_all_telemetry", _load)


def test_checkpoint_exists_after_every_epoch_not_only_at_the_end(tmp_path, fake_telemetry):
    # Training previously saved only after the final epoch, so a hang at epoch 17 of 20
    # threw away 43 minutes of work.
    seen = []
    out = str(tmp_path / "t.pt")

    def spy(epoch, history):
        seen.append((epoch, __import__("os").path.exists(out)))

    train_tokenizer(["a", "b"], out, epochs=3, window=10, batch_size=8, device="cpu", on_epoch=spy)

    assert seen == [(1, True), (2, True), (3, True)], "checkpoint must be on disk from epoch 1"


def test_a_checkpoint_written_mid_training_is_loadable(tmp_path, fake_telemetry):
    out = str(tmp_path / "t.pt")
    loaded = {}

    def spy(epoch, history):
        if epoch == 2:
            model, mean, std, config = load_checkpoint(out)
            loaded["ok"] = config["window"] == 10 and mean is not None

    train_tokenizer(["a"], out, epochs=3, window=10, batch_size=8, device="cpu", on_epoch=spy)

    assert loaded.get("ok") is True


def test_best_validation_checkpoint_is_kept_separately(tmp_path, fake_telemetry):
    # Training loss bottomed at epoch 4 and rose to epoch 20 on the real corpus, so the
    # final-epoch checkpoint is knowingly worse than the best one seen.
    out = str(tmp_path / "t.pt")

    result = train_tokenizer(
        ["a", "b"], out, val_paths=["c"], epochs=4, window=10, batch_size=8, device="cpu"
    )

    best = str(tmp_path / "t.best.pt")
    import os
    assert os.path.exists(best)
    val_losses = [h["val_loss"] for h in result["history"]]
    assert result["best_epoch"] == 1 + val_losses.index(min(val_losses))
    assert result["best_val_loss"] == min(val_losses)


def test_no_best_checkpoint_without_validation_demos(tmp_path, fake_telemetry):
    import os
    out = str(tmp_path / "t.pt")

    result = train_tokenizer(["a"], out, epochs=2, window=10, batch_size=8, device="cpu")

    assert not os.path.exists(str(tmp_path / "t.best.pt"))
    assert result["best_epoch"] is None
