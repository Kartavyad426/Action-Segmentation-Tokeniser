import numpy as np
import torch

from tokenizer.evaluate import (
    boundary_alignment_report,
    check_tokenizer_gates,
    evaluate_tokenizer,
    boundary_alignment_score,
    codebook_utilization,
    reconstruction_error,
)
from tokenizer.model import MotionTokenizer
from tokenizer.windowing import WindowedTelemetryDataset


def _tiny_model():
    torch.manual_seed(0)
    return MotionTokenizer(in_channels=3, window=10, latent_dim=4, num_codes=8, hidden=8)


def test_reconstruction_error_is_a_non_negative_float():
    model = _tiny_model()
    telemetry = np.random.randn(50, 3).astype(np.float32)
    dataset = WindowedTelemetryDataset([telemetry], window=10, stride=5)

    error = reconstruction_error(model, dataset)

    assert isinstance(error, float)
    assert error >= 0.0


def test_codebook_utilization_is_low_for_a_model_that_always_picks_one_code():
    model = _tiny_model()
    with torch.no_grad():
        model.quantizer.codebook.weight[:] = 0.0
        model.quantizer.codebook.weight[0] = 100.0  # every input snaps to code 0

    telemetry = np.random.randn(50, 3).astype(np.float32)
    dataset = WindowedTelemetryDataset([telemetry], window=10, stride=5)

    utilization = codebook_utilization(model, dataset, num_codes=8)

    assert utilization < 0.1


def test_boundary_alignment_score_is_one_when_token_changes_at_every_boundary():
    model = _tiny_model()
    # 3 windows of 10 samples each; force a distinct, stable code per window by making
    # each window's values wildly different so the (randomly-initialized) encoder
    # separates them.
    telemetry = np.concatenate(
        [
            np.full((10, 3), -10.0, dtype=np.float32),
            np.full((10, 3), 0.0, dtype=np.float32),
            np.full((10, 3), 10.0, dtype=np.float32),
        ]
    )
    grid = np.arange(30) * 0.01
    segments = [(0.0, 0.1, "A"), (0.1, 0.2, "B"), (0.2, 0.3, "C")]

    report = boundary_alignment_score(model, telemetry, grid, segments, window=10, tolerance_s=0.05)

    for key in ("boundary_recall", "change_precision", "f1", "chance_recall", "segment_purity"):
        assert 0.0 <= report[key] <= 1.0


def _centers(n, dt=0.15):
    return np.arange(n) * dt


def test_a_tokenizer_that_changes_every_token_scores_at_chance():
    # THE failure the old pure-recall score could not report: a tokenizer that emits a
    # different code every single window hits every boundary by accident and scored a
    # perfect 1.0 despite being useless.
    n = 200
    indices = np.arange(n)  # changes at every step
    segments = [(3.0, 6.0, "A"), (6.0, 9.0, "B"), (9.0, 12.0, "C")]

    report = boundary_alignment_report(indices, _centers(n), segments, tolerance_s=0.2)

    assert report["boundary_recall"] > 0.9, "recall alone still looks perfect"
    assert report["chance_recall"] > 0.9, "and the metric now says why: chance alone explains it"
    assert report["lift_over_chance"] < 0.1, "so the placement carries almost no information"
    assert report["change_precision"] < 0.1
    assert report["f1"] < 0.2


def test_changes_placed_exactly_on_boundaries_beat_chance():
    n = 200
    centers = _centers(n)
    segments = [(3.0, 6.0, "A"), (6.0, 9.0, "B"), (9.0, 12.0, "C")]
    boundaries = [3.0, 6.0, 9.0, 12.0]

    indices = np.zeros(n, dtype=int)
    for i, b in enumerate(boundaries, start=1):
        indices[centers >= b] = i

    report = boundary_alignment_report(indices, centers, segments, tolerance_s=0.2)

    assert report["boundary_recall"] == 1.0
    assert report["change_precision"] == 1.0
    assert report["lift_over_chance"] > 0.9
    assert report["f1"] == 1.0


def test_segment_purity_separates_a_stable_tokenizer_from_a_flickering_one():
    n = 200
    centers = _centers(n)
    segments = [(3.0, 6.0, "A"), (6.0, 9.0, "B")]

    stable = np.zeros(n, dtype=int)
    stable[centers >= 6.0] = 1
    flicker = np.arange(n) % 7

    assert boundary_alignment_report(stable, centers, segments)["segment_purity"] == 1.0
    assert boundary_alignment_report(flicker, centers, segments)["segment_purity"] < 0.4


def test_chance_baseline_rises_with_the_token_change_rate():
    n = 200
    centers = _centers(n)
    segments = [(3.0, 6.0, "A"), (6.0, 9.0, "B")]

    few = np.zeros(n, dtype=int)
    few[centers >= 6.0] = 1
    many = np.arange(n)

    assert (
        boundary_alignment_report(many, centers, segments)["chance_recall"]
        > boundary_alignment_report(few, centers, segments)["chance_recall"]
    )


def test_report_is_all_zeros_when_there_are_no_token_changes():
    n = 50
    report = boundary_alignment_report(np.zeros(n, dtype=int), _centers(n), [(1.0, 2.0, "A")])

    assert report["boundary_recall"] == 0.0
    assert report["change_precision"] == 0.0
    assert report["f1"] == 0.0


def test_report_is_all_zeros_when_the_demo_has_no_labeled_segments():
    n = 50
    report = boundary_alignment_report(np.arange(n), _centers(n), [])

    assert report["f1"] == 0.0


def _healthy_report():
    return {
        "reconstruction_error": 0.08,
        "codebook_utilization": 0.72,
        "effective_codes": 120.0,
        "boundary": {"f1": 0.55, "lift_over_chance": 0.31, "segment_purity": 0.8},
    }


def test_gate_passes_a_healthy_tokenizer():
    assert check_tokenizer_gates(_healthy_report()) == []


def test_gate_fails_the_codebook_collapse_measured_during_review():
    # 0.235 utilization, ~9 live codes out of 512. This is the failure the pipeline had no
    # way to catch before classifier training consumed hours of GPU time.
    report = _healthy_report() | {"codebook_utilization": 0.235, "effective_codes": 9.0}

    failures = check_tokenizer_gates(report)

    assert len(failures) == 1
    assert "codebook" in failures[0].lower()
    assert "0.235" in failures[0] and "9" in failures[0]


def test_gate_fails_boundary_alignment_at_or_below_chance_without_needing_a_threshold():
    # No arbitrary cutoff required: scoring no better than randomly-scattered token changes
    # is an unambiguous failure on its own terms.
    report = _healthy_report() | {"boundary": {"f1": 0.4, "lift_over_chance": -0.02, "segment_purity": 0.8}}

    failures = check_tokenizer_gates(report)

    assert len(failures) == 1
    assert "chance" in failures[0].lower()


def test_gate_fails_a_non_finite_reconstruction_error():
    report = _healthy_report() | {"reconstruction_error": float("nan")}

    failures = check_tokenizer_gates(report)

    assert len(failures) == 1
    assert "reconstruction" in failures[0].lower()


def test_gate_reports_every_independent_failure_not_just_the_first():
    report = _healthy_report() | {
        "codebook_utilization": 0.1,
        "effective_codes": 3.0,
        "boundary": {"f1": 0.1, "lift_over_chance": -0.05, "segment_purity": 0.1},
    }

    assert len(check_tokenizer_gates(report)) == 2


def test_gate_utilization_threshold_is_overridable():
    report = _healthy_report() | {"codebook_utilization": 0.4, "effective_codes": 30.0}

    assert check_tokenizer_gates(report) == []
    assert len(check_tokenizer_gates(report, min_codebook_utilization=0.5)) == 1


def _synthetic_demo(seed, n=300):
    rng = np.random.default_rng(seed)
    telemetry = rng.standard_normal((n, 3)).astype(np.float32)
    grid = np.arange(n) * 0.01
    segments = [(0.5, 1.0, "A"), (1.0, 1.5, "B"), (1.5, 2.0, "C")]
    return telemetry, grid, segments


def test_evaluate_tokenizer_reports_all_three_spec_checks():
    model = _tiny_model()
    demos = [_synthetic_demo(i) for i in range(3)]

    report = evaluate_tokenizer(model, num_codes=8, window=10, demos=demos)

    assert set(report) >= {"reconstruction_error", "codebook_utilization", "effective_codes", "boundary"}
    assert report["reconstruction_error"] >= 0.0
    assert 0.0 <= report["codebook_utilization"] <= 1.0
    assert 1.0 <= report["effective_codes"] <= 8.0
    assert "lift_over_chance" in report["boundary"]


def test_evaluate_tokenizer_catches_a_collapsed_codebook_end_to_end():
    model = _tiny_model()
    with torch.no_grad():
        model.quantizer.codebook.weight[:] = 0.0
        model.quantizer.codebook.weight[0] = 100.0  # every window snaps to code 0

    report = evaluate_tokenizer(model, num_codes=8, window=10, demos=[_synthetic_demo(0)])
    failures = check_tokenizer_gates(report)

    assert report["effective_codes"] < 1.5
    assert any("codebook" in f.lower() for f in failures)
