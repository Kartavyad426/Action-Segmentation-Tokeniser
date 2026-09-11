"""Timing numbers are meaningless without the power state they were measured in."""

from unittest.mock import patch

from temporal_classifier.provenance import (
    format_gpu_state,
    gpu_state,
    runtime_differences,
)

_SMI = "NVIDIA RTX PRO 1000, 580.95.05, P1, 1380, 3090, 24.83, 35.00, 35.00, 68, 54\n"
_THROTTLED = "NVIDIA RTX PRO 1000, 580.95.05, P8, 180, 3090, 8.20, 15.00, 35.00, 100, 41\n"


def _run_ok(out):
    return patch("subprocess.check_output", return_value=out)


def test_gpu_state_captures_clocks_and_power_limit_not_just_utilization():
    # utilization.gpu reads 100% while the card is clamped to 180MHz, so it cannot
    # distinguish "working" from "working 25x too slowly". pstate and clocks can.
    with _run_ok(_SMI):
        state = gpu_state()

    assert state["pstate"] == "P1"
    assert state["clocks_sm_mhz"] == 1380
    assert state["clocks_max_sm_mhz"] == 3090
    assert state["enforced_power_limit_w"] == 35.0
    assert state["default_power_limit_w"] == 35.0


def test_gpu_state_survives_a_machine_without_nvidia_smi():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        state = gpu_state()

    assert state["available"] is False
    assert "pstate" not in state or state.get("pstate") is None


def test_runtime_differences_flags_a_power_state_change():
    with _run_ok(_SMI):
        healthy = gpu_state()
    with _run_ok(_THROTTLED):
        throttled = gpu_state()

    diffs = runtime_differences(healthy, throttled)

    assert any("clocks_sm_mhz" in d for d in diffs)
    assert any("enforced_power_limit_w" in d for d in diffs)


def test_runtime_differences_is_empty_for_the_same_state():
    with _run_ok(_SMI):
        a = gpu_state()
    with _run_ok(_SMI):
        b = gpu_state()

    assert runtime_differences(a, b) == []


def test_runtime_differences_ignores_instantaneous_readings():
    # power.draw, utilization and temperature fluctuate second to second; flagging them
    # would make every comparison noisy and train the reader to ignore the warning.
    with _run_ok(_SMI):
        a = gpu_state()
    b = dict(a, power_draw_w=31.0, utilization_pct=12, temperature_c=70)

    assert runtime_differences(a, b) == []


def test_format_gpu_state_is_one_readable_line():
    with _run_ok(_THROTTLED):
        text = format_gpu_state(gpu_state())

    assert "P8" in text and "180" in text and "15.0" in text
