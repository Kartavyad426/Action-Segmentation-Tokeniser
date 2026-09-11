"""A hang or a kill must never cost the whole tokenizer training run."""

import numpy as np
import pytest
import torch

from tokenizer.train import load_checkpoint, select_best_epoch, train_tokenizer


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


def test_no_separate_best_checkpoint_is_written(tmp_path, fake_telemetry):
    # A "best" checkpoint implies a selection rule. There isn't one any more -- the run keeps
    # its final epoch, and what a rule *would* have chosen is reported instead.
    import os
    out = str(tmp_path / "t.pt")

    result = train_tokenizer(
        ["a", "b"], out, val_paths=["c"], epochs=4, window=10, batch_size=8, device="cpu"
    )

    assert not os.path.exists(str(tmp_path / "t.best.pt"))
    assert result["best_checkpoint"] is None
    assert result["best_epoch"] is not None, "still reported as a metric"


def test_no_selection_metric_without_validation_demos(tmp_path, fake_telemetry):
    result = train_tokenizer(["a"], str(tmp_path / "t.pt"), epochs=2, window=10,
                             batch_size=8, device="cpu")

    assert result["best_epoch"] is None
    assert result["selected_epoch"] == 2


def test_history_records_codebook_utilization_per_epoch(tmp_path, fake_telemetry):
    result = train_tokenizer(
        ["a", "b"], str(tmp_path / "t.pt"), val_paths=["c"], epochs=3,
        window=10, num_codes=8, batch_size=8, device="cpu",
    )

    assert all(0.0 <= h["val_utilization"] <= 1.0 for h in result["history"])


def test_best_checkpoint_is_chosen_by_codebook_utilization_not_by_loss(tmp_path, fake_telemetry):
    # Measured on real data: the lowest-loss epoch had a collapsed codebook (8 of 512
    # effective codes, gate FAIL) while a higher-loss epoch had 136 (PASS). Total VQ-VAE
    # loss falls when the encoder collapses onto a few codes, because the commitment and
    # codebook terms shrink -- so selecting on loss selects for collapse.
    out = str(tmp_path / "t.pt")
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.0, "val_utilization": 0.80},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.0, "val_utilization": 0.20},
    ]

    assert select_best_epoch(history) == 1, "must not pick the low-loss collapsed epoch"


def test_select_best_epoch_breaks_ties_on_loss():
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.0, "val_utilization": 0.70},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.0, "val_utilization": 0.70},
    ]

    assert select_best_epoch(history) == 2


def test_select_best_epoch_returns_none_without_validation():
    assert select_best_epoch([{"epoch": 1, "train_loss": 1.0, "val_loss": None,
                               "val_utilization": None}]) is None


def test_selection_prefers_low_loss_among_epochs_that_clear_the_utilization_floor():
    # Utilization alone is gameable: a tokenizer assigning codes at random maxes out
    # entropy while carrying no information, the same trap as pure boundary recall. So the
    # floor rejects collapse, and reconstruction loss discriminates among healthy epochs.
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.0, "val_utilization": 0.99},  # noisy
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.0, "val_utilization": 0.60},  # healthy
        {"epoch": 3, "train_loss": 0.4, "val_loss": 0.4, "val_utilization": 0.20},  # collapsed
    ]

    assert select_best_epoch(history, min_utilization=0.35) == 2


def test_selection_falls_back_to_the_least_collapsed_epoch_when_none_clear_the_floor():
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.0, "val_utilization": 0.30},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.5, "val_utilization": 0.10},
    ]

    assert select_best_epoch(history, min_utilization=0.35) == 1


def test_selection_uses_reconstruction_error_not_total_loss():
    # Total VQ-VAE loss = recon + codebook + commitment. The last two GROW as codes spread
    # out of their collapsed initialization, so "lowest total loss above the floor"
    # systematically picks the least-developed tokenizer that scrapes past it. Measured: it
    # chose epoch 4 (utilization 0.510, ~25 live codes) over epoch 20 (0.762, ~100).
    # Reconstruction error is the uncontaminated signal.
    history = [
        {"epoch": 4,  "train_loss": 1.0, "val_loss": 1.1199, "val_recon_loss": 0.62,
         "val_utilization": 0.510},
        {"epoch": 20, "train_loss": 1.0, "val_loss": 1.3469, "val_recon_loss": 0.35,
         "val_utilization": 0.762},
    ]

    assert select_best_epoch(history, min_utilization=0.35) == 20


def test_history_records_reconstruction_error_separately(tmp_path, fake_telemetry):
    result = train_tokenizer(
        ["a", "b"], str(tmp_path / "t.pt"), val_paths=["c"], epochs=2,
        window=10, num_codes=8, batch_size=16, device="cpu", verbose=False,
    )

    for h in result["history"]:
        assert h["val_recon_loss"] is not None
        # total loss carries the VQ terms on top of reconstruction
        assert h["val_loss"] >= h["val_recon_loss"]


def test_selection_falls_back_to_total_loss_for_histories_without_recon():
    # older runs recorded no recon term
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 2.0, "val_utilization": 0.60},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.0, "val_utilization": 0.60},
    ]

    assert select_best_epoch(history, min_utilization=0.35) == 2


def test_the_checkpoint_is_the_final_epoch_not_a_selected_one(tmp_path, fake_telemetry):
    # Selection is deliberately not applied. Three different selection rules were tried in
    # one session and each one changed which tokenizer a run produced, silently invalidating
    # comparisons against earlier runs. The final epoch is deterministic and needs no rule.
    out = str(tmp_path / "t.pt")

    result = train_tokenizer(
        ["a", "b"], out, val_paths=["c"], epochs=4, window=10, num_codes=8,
        batch_size=16, device="cpu", verbose=False,
    )

    assert load_checkpoint(out)[3]["window"] == 10
    import torch
    assert torch.load(out, weights_only=False)["epoch"] == 4, "must be the last epoch"


def test_selection_metrics_are_still_reported_just_not_acted_on(tmp_path, fake_telemetry):
    result = train_tokenizer(
        ["a", "b"], str(tmp_path / "t.pt"), val_paths=["c"], epochs=4, window=10,
        num_codes=8, batch_size=16, device="cpu", verbose=False,
    )

    # the metrics remain, for a human to read
    assert all(h["val_utilization"] is not None for h in result["history"])
    assert all(h["val_recon_loss"] is not None for h in result["history"])
    # and what a selection rule *would* have chosen is reported, without being used
    assert result["best_epoch"] == select_best_epoch(result["history"])
    assert result["selected_epoch"] == 4
