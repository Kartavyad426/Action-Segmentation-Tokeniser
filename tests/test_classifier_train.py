import os
from unittest.mock import patch

import numpy as np
import pytest

from enrichment.vision_features import VISION_FEATURE_DIM
from temporal_classifier.train import run_training
from tokenizer.train import train_tokenizer
from tokenizer.windowing import split_available_demos

requires_data = pytest.mark.skipif(
    not os.path.exists("data/reassemble"), reason="REASSEMBLE demos not downloaded"
)


@requires_data
def test_run_training_produces_a_val_f1_score(tmp_path):
    train_paths, val_paths = split_available_demos()
    all_paths = (train_paths + val_paths)[:4]
    assert len(all_paths) >= 2, "need at least 2 downloaded demos for a train/val smoke test"
    train_subset, val_subset = all_paths[:-1], all_paths[-1:]

    tokenizer_ckpt = str(tmp_path / "tokenizer.pt")
    train_tokenizer(train_subset, tokenizer_ckpt, epochs=2, batch_size=8, device="cpu")

    result = run_training(
        tokenizer_ckpt, train_subset, val_subset, epochs=2, channels=8, num_layers=2, num_stages=2, device="cpu"
    )

    assert "val_f1" in result
    assert 0.0 <= result["val_f1"] <= 1.0
    assert "background" in result["vocab"]


def _fake_extract_frame_features(h5_path, frame_times, encoder, camera_key="hama1", device="cuda"):
    # run_training(..., device="cpu") below must thread that device all the way through
    # to this call. If it silently falls back to the "cuda" default (Finding 1's bug),
    # catch it here instead of only in a real GPU/CPU tensor-mismatch crash.
    assert device == "cpu", f"expected device='cpu' to be threaded through to extract_frame_features, got {device!r}"
    return np.zeros((len(frame_times), VISION_FEATURE_DIM), dtype=np.float32)


@requires_data
def test_run_training_with_vision_fusion_completes_on_cpu_and_matches_telemetry_only_vocab(tmp_path):
    # Regression test for a device-propagation bug where _prepare_split failed to pass
    # `device` through to extract_frame_features, which then defaulted to "cuda" and
    # crashed on a CPU-only run_training(..., use_vision=True) call. Mocks out the real
    # DINOv2 feature extraction (no GPU / model download needed) but exercises the real
    # run_training(..., use_vision=True, device="cpu") code path end to end.
    train_paths, val_paths = split_available_demos()
    all_paths = (train_paths + val_paths)[:4]
    assert len(all_paths) >= 2, "need at least 2 downloaded demos for a train/val smoke test"
    train_subset, val_subset = all_paths[:-1], all_paths[-1:]

    tokenizer_ckpt = str(tmp_path / "tokenizer.pt")
    train_tokenizer(train_subset, tokenizer_ckpt, epochs=2, batch_size=8, device="cpu")

    telemetry_only = run_training(
        tokenizer_ckpt, train_subset, val_subset, epochs=2, channels=8, num_layers=2, num_stages=2, device="cpu"
    )

    with patch("temporal_classifier.train.extract_frame_features", side_effect=_fake_extract_frame_features):
        result = run_training(
            tokenizer_ckpt,
            train_subset,
            val_subset,
            vocab=telemetry_only["vocab"],
            epochs=2,
            channels=8,
            num_layers=2,
            num_stages=2,
            device="cpu",
            use_vision=True,
            vision_encoder=object(),  # never actually called, since extract_frame_features is mocked
        )

    assert result["fusion"] is not None
    assert 0.0 <= result["val_f1"] <= 1.0
    assert result["vocab"] == telemetry_only["vocab"]
