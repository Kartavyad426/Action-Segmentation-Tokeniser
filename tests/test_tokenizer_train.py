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
