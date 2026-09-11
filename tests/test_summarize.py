import json
import os

import pytest

from temporal_classifier.summarize import collect_runs, format_summary, summarize


def _run(tmp_path, name, arms, fingerprint="F", tok_seed=0, cls_seed=0):
    d = tmp_path / name
    d.mkdir()
    json.dump({"fingerprint": fingerprint, "tokenizer_hp": {"seed": tok_seed},
               "classifier_hp": {"seed": cls_seed}, "split_scored": "validation"},
              open(d / "config.json", "w"))
    json.dump({"fingerprint": fingerprint,
               "arms": {k: {"val_f1": v} for k, v in arms.items()}},
              open(d / "results.json", "w"))
    return d


def test_collect_runs_skips_runs_without_results(tmp_path):
    _run(tmp_path, "a", {"telemetry_only": 0.84})
    (tmp_path / "b").mkdir()  # aborted run, no results.json

    assert len(collect_runs(str(tmp_path))) == 1


def test_summarize_reports_mean_and_spread_per_arm(tmp_path):
    _run(tmp_path, "a", {"telemetry_only": 0.80, "vision_nearest": 0.82}, cls_seed=0)
    _run(tmp_path, "b", {"telemetry_only": 0.84, "vision_nearest": 0.86}, cls_seed=1)

    summary = summarize(collect_runs(str(tmp_path)))

    assert summary["telemetry_only"]["n"] == 2
    assert summary["telemetry_only"]["mean"] == pytest.approx(0.82)
    assert summary["telemetry_only"]["min"] == pytest.approx(0.80)
    assert summary["telemetry_only"]["max"] == pytest.approx(0.84)
    assert summary["telemetry_only"]["std"] == pytest.approx(0.02)


def test_summarize_reports_the_paired_delta_not_just_arm_means(tmp_path):
    # The deliverable is "does vision help", so the delta must be computed within each run
    # and then averaged -- differencing two arm means discards the pairing and overstates
    # confidence when runs vary a lot.
    _run(tmp_path, "a", {"telemetry_only": 0.80, "vision_nearest": 0.81}, cls_seed=0)
    _run(tmp_path, "b", {"telemetry_only": 0.90, "vision_nearest": 0.91}, cls_seed=1)

    summary = summarize(collect_runs(str(tmp_path)))

    assert summary["vision_nearest"]["delta_mean"] == pytest.approx(0.01)
    assert summary["vision_nearest"]["delta_std"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_ignores_a_run_missing_the_control_arm_for_delta(tmp_path):
    _run(tmp_path, "a", {"vision_nearest": 0.81})

    summary = summarize(collect_runs(str(tmp_path)))

    assert summary["vision_nearest"]["delta_mean"] is None


def test_format_summary_is_readable_and_mentions_every_arm(tmp_path):
    _run(tmp_path, "a", {"telemetry_only": 0.80, "vision_nearest": 0.82})

    text = format_summary(summarize(collect_runs(str(tmp_path))))

    assert "telemetry_only" in text and "vision_nearest" in text
    assert "0.80" in text
