import json
import os

import pytest

from temporal_classifier.plots import (
    classifier_series,
    parse_tokenizer_log,
    plot_run,
    tokenizer_series,
)


def _history(n=6):
    return [
        {"epoch": i, "train_loss": 2.0 - i * 0.1, "val_loss": 2.1 - i * 0.05}
        for i in range(1, n + 1)
    ]


def test_tokenizer_series_splits_epochs_from_losses():
    epochs, train, val = tokenizer_series(_history(3))

    assert epochs == [1, 2, 3]
    assert train == pytest.approx([1.9, 1.8, 1.7])
    assert val == pytest.approx([2.05, 2.0, 1.95])


def test_tokenizer_series_tolerates_a_run_without_validation():
    history = [{"epoch": 1, "train_loss": 1.0, "val_loss": None}]

    epochs, train, val = tokenizer_series(history)

    assert val == []


def test_classifier_series_keeps_only_the_epochs_that_were_scored():
    # val F1 is measured every eval_every epochs, so the F1 series is sparser than the loss
    # series and must carry its own epoch axis rather than being zipped against all epochs.
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_f1": None},
        {"epoch": 2, "train_loss": 0.8, "val_f1": 0.30},
        {"epoch": 3, "train_loss": 0.7, "val_f1": None},
        {"epoch": 4, "train_loss": 0.6, "val_f1": 0.45},
    ]

    epochs, loss, f1_epochs, f1 = classifier_series(history)

    assert epochs == [1, 2, 3, 4]
    assert loss == pytest.approx([1.0, 0.8, 0.7, 0.6])
    assert f1_epochs == [2, 4]
    assert f1 == pytest.approx([0.30, 0.45])


def _write_run(tmp_path, with_results=True):
    d = tmp_path / "run"
    d.mkdir()
    json.dump({"fingerprint": "F", "history": _history(), "best_epoch": 4, "best_val_loss": 1.9},
              open(d / "tokenizer_history.json", "w"))
    json.dump({"fingerprint": "F", "reconstruction_error": 0.35, "codebook_utilization": 0.78,
               "effective_codes": 136.0, "n_demos": 22,
               "boundary": {"f1": 0.2, "boundary_recall": 0.64, "change_precision": 0.12,
                            "chance_recall": 0.49, "lift_over_chance": 0.15, "segment_purity": 0.47}},
              open(d / "tokenizer_report.json", "w"))
    if with_results:
        json.dump({"fingerprint": "F", "arms": {
            "telemetry_only": {"val_f1": 0.41, "seconds": 120, "history": [
                {"epoch": 1, "train_loss": 1.2, "val_f1": None},
                {"epoch": 2, "train_loss": 0.9, "val_f1": 0.41}]},
            "vision_nearest": {"val_f1": 0.44, "seconds": 300, "history": [
                {"epoch": 1, "train_loss": 1.1, "val_f1": None},
                {"epoch": 2, "train_loss": 0.8, "val_f1": 0.44}]},
        }}, open(d / "results.json", "w"))
    return d


def test_plot_run_writes_a_figure_for_each_available_artifact(tmp_path):
    d = _write_run(tmp_path)

    written = plot_run(str(d))

    names = {os.path.basename(p) for p in written}
    assert "tokenizer_training.png" in names
    assert "classifier_training.png" in names
    assert "f1_comparison.png" in names
    for p in written:
        assert os.path.getsize(p) > 1000, f"{p} looks empty"


def test_plot_run_works_on_a_run_that_has_not_reached_the_arms_yet(tmp_path):
    # A run killed during tokenizer training should still be plottable.
    d = _write_run(tmp_path, with_results=False)

    written = plot_run(str(d))

    names = {os.path.basename(p) for p in written}
    assert "tokenizer_training.png" in names
    assert "f1_comparison.png" not in names


def test_plot_run_on_an_empty_directory_writes_nothing_and_does_not_raise(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()

    assert plot_run(str(d)) == []


def test_tokenizer_history_can_be_recovered_from_the_run_log(tmp_path):
    # tokenizer_history.json is written as the run progresses, but a run from before that
    # change -- or one killed between epochs -- still has its curve in the log.
    log = tmp_path / "run.log"
    log.write_text(
        "10:45:21  training tokenizer on 89 demos, 20 epochs\n"
        "10:52:10    tokenizer epoch   1/20  train 1.67035  val 1.47374\n"
        "10:54:40    tokenizer epoch   2/20  train 1.39899  val 1.20211\n"
        "10:57:09    tokenizer epoch   3/20  train 1.31235\n"
        "11:00:00  tokenizer ready at runs/x/tokenizer_checkpoint.pt\n"
    )

    history = parse_tokenizer_log(str(log))

    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert history[0]["train_loss"] == pytest.approx(1.67035)
    assert history[1]["val_loss"] == pytest.approx(1.20211)
    assert history[2]["val_loss"] is None


def test_plot_run_falls_back_to_the_log_when_history_json_is_missing(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "run.log").write_text(
        "10:52:10    tokenizer epoch   1/20  train 1.67035  val 1.47374\n"
        "10:54:40    tokenizer epoch   2/20  train 1.39899  val 1.20211\n"
    )

    written = plot_run(str(d))

    assert [os.path.basename(p) for p in written] == ["tokenizer_training.png"]
