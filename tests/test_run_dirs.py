"""Every run gets its own directory; nothing from a previous run is ever overwritten."""

import json
import os
from datetime import datetime

import pytest

from temporal_classifier.compare import RunPaths, adopt_previous_run, new_run, write_run_config


def test_each_run_gets_its_own_timestamped_directory(tmp_path):
    a = new_run(str(tmp_path), now=datetime(2026, 9, 10, 14, 30, 0))
    b = new_run(str(tmp_path), now=datetime(2026, 9, 10, 14, 30, 1))

    assert a.dir != b.dir
    assert os.path.isdir(a.dir) and os.path.isdir(b.dir)
    assert "20260910-143000" in a.dir


def test_a_second_run_in_the_same_second_does_not_clobber_the_first(tmp_path):
    same = datetime(2026, 9, 10, 14, 30, 0)
    a = new_run(str(tmp_path), now=same)
    (open(os.path.join(a.dir, "marker"), "w")).write("first")

    b = new_run(str(tmp_path), now=same)

    assert a.dir != b.dir
    assert os.path.exists(os.path.join(a.dir, "marker")), "the first run's files must survive"


def test_the_checkpoint_lives_inside_the_run_directory(tmp_path):
    run = new_run(str(tmp_path))

    assert os.path.dirname(run.checkpoint) == run.dir
    for p in (run.log, run.config, run.results, run.tokenizer_report):
        assert os.path.dirname(p) == run.dir


def test_latest_points_at_the_newest_run(tmp_path):
    new_run(str(tmp_path), now=datetime(2026, 9, 10, 14, 0, 0))
    b = new_run(str(tmp_path), now=datetime(2026, 9, 10, 15, 0, 0))

    assert os.path.realpath(os.path.join(str(tmp_path), "latest")) == os.path.realpath(b.dir)


def test_config_records_everything_needed_to_reproduce_the_run(tmp_path):
    run = new_run(str(tmp_path))

    written = write_run_config(
        run, eval_on="val", fingerprint="abc123",
        tokenizer_hp={"num_codes": 512}, classifier_hp={"epochs": 30},
        splits={"train": ["a.h5"], "val": ["b.h5"], "test": ["c.h5"]},
    )

    saved = json.loads(open(run.config).read())
    assert saved == written
    assert saved["fingerprint"] == "abc123"
    assert saved["tokenizer_hp"]["num_codes"] == 512
    assert saved["splits"]["counts"] == {"train": 1, "val": 1, "test": 1}
    # the actual demo membership, not just counts -- a split that shifts must be detectable
    assert saved["splits"]["train"] == ["a.h5"]
    for key in ("started_at", "git_commit", "torch_version", "python_version", "command"):
        assert key in saved


def test_adopting_a_previous_run_reuses_its_checkpoint_when_the_fingerprint_matches(tmp_path):
    import torch
    old = new_run(str(tmp_path), now=datetime(2026, 9, 10, 10, 0, 0))
    torch.save({"config": {"fingerprint": "F"}}, old.checkpoint)
    json.dump({"fingerprint": "F", "arms": {"telemetry_only": {"val_f1": 0.5}}}, open(old.results, "w"))

    new = new_run(str(tmp_path), now=datetime(2026, 9, 10, 11, 0, 0))
    adopted = adopt_previous_run(new, old.dir, fingerprint="F")

    assert adopted is True
    assert os.path.exists(new.checkpoint), "checkpoint copied into the new run dir"
    assert json.loads(open(new.results).read())["arms"]["telemetry_only"]["val_f1"] == 0.5
    assert os.path.exists(old.checkpoint), "the previous run must be left untouched"


def test_adopting_refuses_a_previous_run_with_a_different_fingerprint(tmp_path):
    import torch
    old = new_run(str(tmp_path), now=datetime(2026, 9, 10, 10, 0, 0))
    torch.save({"config": {"fingerprint": "OLD"}}, old.checkpoint)

    new = new_run(str(tmp_path), now=datetime(2026, 9, 10, 11, 0, 0))
    adopted = adopt_previous_run(new, old.dir, fingerprint="NEW")

    assert adopted is False
    assert not os.path.exists(new.checkpoint)
