"""Guards against silently reusing state from a previous, differently-configured run."""

import json

import numpy as np
import pytest

from temporal_classifier.compare import ResultLog, resolve_tokenizer
from tokenizer.train import tokenizer_fingerprint


def test_fingerprint_changes_when_the_training_demos_change():
    a = tokenizer_fingerprint(["a.h5", "b.h5"], window=15, num_codes=512)
    b = tokenizer_fingerprint(["a.h5", "c.h5"], window=15, num_codes=512)

    assert a != b


def test_fingerprint_changes_when_a_hyperparameter_changes():
    a = tokenizer_fingerprint(["a.h5"], window=15, num_codes=512)
    b = tokenizer_fingerprint(["a.h5"], window=15, num_codes=64)

    assert a != b


def test_fingerprint_is_stable_and_order_independent():
    a = tokenizer_fingerprint(["a.h5", "b.h5"], window=15, num_codes=512)
    b = tokenizer_fingerprint(["b.h5", "a.h5"], window=15, num_codes=512)

    assert a == b


def _write_ckpt(path, fingerprint):
    import torch
    torch.save({"model_state": {}, "mean": np.zeros(1), "std": np.ones(1),
                "config": {"fingerprint": fingerprint}}, str(path))


def test_a_checkpoint_from_a_different_config_is_never_reused(tmp_path):
    ckpt = tmp_path / "t.pt"
    _write_ckpt(ckpt, "OLDFINGERPRINT")

    decision = resolve_tokenizer(str(ckpt), expected_fingerprint="NEWFINGERPRINT", keep=True)

    assert decision.retrain is True
    assert not ckpt.exists(), "the stale checkpoint must be removed, not left to be reloaded"
    assert "fingerprint" in decision.reason.lower()


def test_a_matching_checkpoint_is_reused_when_keep_is_requested(tmp_path):
    ckpt = tmp_path / "t.pt"
    _write_ckpt(ckpt, "SAME")

    decision = resolve_tokenizer(str(ckpt), expected_fingerprint="SAME", keep=True)

    assert decision.retrain is False
    assert ckpt.exists()


def test_keep_is_ignored_when_no_checkpoint_exists(tmp_path):
    decision = resolve_tokenizer(str(tmp_path / "absent.pt"), expected_fingerprint="X", keep=True)

    assert decision.retrain is True


def test_default_always_retrains_even_if_a_matching_checkpoint_exists(tmp_path):
    ckpt = tmp_path / "t.pt"
    _write_ckpt(ckpt, "SAME")

    decision = resolve_tokenizer(str(ckpt), expected_fingerprint="SAME", keep=False)

    assert decision.retrain is True
    assert not ckpt.exists()


def test_result_log_persists_each_arm_as_it_completes(tmp_path):
    # A finished arm must survive the process dying before the other arms complete.
    log = ResultLog(str(tmp_path / "results.json"))
    log.record("telemetry_only", {"val_f1": 0.42}, fingerprint="F")

    reloaded = ResultLog(str(tmp_path / "results.json"))

    assert reloaded.get("telemetry_only")["val_f1"] == 0.42
    assert json.loads((tmp_path / "results.json").read_text())["arms"]["telemetry_only"]["val_f1"] == 0.42


def test_result_log_discards_arms_from_a_different_tokenizer(tmp_path):
    # Arms are only comparable if they share one tokenizer; a retrained tokenizer
    # invalidates every previously recorded arm.
    log = ResultLog(str(tmp_path / "results.json"))
    log.record("telemetry_only", {"val_f1": 0.42}, fingerprint="OLD")

    reloaded = ResultLog(str(tmp_path / "results.json"), fingerprint="NEW")

    assert reloaded.get("telemetry_only") is None


def test_result_log_keeps_arms_from_the_same_tokenizer(tmp_path):
    log = ResultLog(str(tmp_path / "results.json"), fingerprint="SAME")
    log.record("telemetry_only", {"val_f1": 0.42}, fingerprint="SAME")

    reloaded = ResultLog(str(tmp_path / "results.json"), fingerprint="SAME")

    assert reloaded.get("telemetry_only")["val_f1"] == 0.42
