import os

import pytest

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
