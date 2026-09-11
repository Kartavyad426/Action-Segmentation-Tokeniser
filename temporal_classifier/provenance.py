"""Runtime facts that change what a measurement means, recorded alongside every number.

Motivating incident: identical feature-extraction work on the same demo file took 601.7 s
and 24.0 s in the same run, because the laptop was on battery for the first and on AC for
the second (GPU clamped to 180 MHz / 15 W, versus 1380 MHz / 35 W). Nothing in the run
record distinguished the two, so `wall_seconds` was a number that silently meant two things
25x apart.

`utilization.gpu` does not help: it reads 100% while the card is throttled, because a kernel
is resident even when it is running at a fraction of speed. `pstate` and `clocks.sm` are what
actually say whether the hardware is working at full rate.

These facts are deliberately kept OUT of the tokenizer fingerprint that gates checkpoint
reuse. A checkpoint trained at 15 W is perfectly valid at 35 W; the power state should be
reported on a mismatch, not treated as grounds for refusing the work.
"""

import subprocess

_FIELDS = (
    "name,driver_version,pstate,clocks.sm,clocks.max.sm,power.draw,"
    "enforced.power.limit,power.default_limit,utilization.gpu,temperature.gpu"
)

# Values that legitimately fluctuate second to second. Flagging them would make every
# comparison noisy and teach the reader to ignore the warning.
_VOLATILE = {"power_draw_w", "utilization_pct", "temperature_c", "available"}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gpu_state() -> dict:
    """A snapshot of what the GPU is actually capable of right now."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=15,
        )
    except Exception:
        return {"available": False}

    parts = [p.strip() for p in raw.strip().splitlines()[0].split(",")]
    if len(parts) < 10:
        return {"available": False}

    clocks, clocks_max = _to_float(parts[3]), _to_float(parts[4])
    state = {
        "available": True,
        "name": parts[0],
        "driver_version": parts[1],
        "pstate": parts[2],
        "clocks_sm_mhz": int(clocks) if clocks is not None else None,
        "clocks_max_sm_mhz": int(clocks_max) if clocks_max is not None else None,
        "power_draw_w": _to_float(parts[5]),
        "enforced_power_limit_w": _to_float(parts[6]),
        "default_power_limit_w": _to_float(parts[7]),
        "utilization_pct": _to_float(parts[8]),
        "temperature_c": _to_float(parts[9]),
    }
    if clocks and clocks_max:
        state["clock_fraction_of_max"] = round(clocks / clocks_max, 3)
    return state


def runtime_differences(before: dict, after: dict) -> list[str]:
    """Stable fields that changed between two snapshots, as readable strings.

    Reported, never fatal: this is the broader provenance record, not the key that decides
    whether a checkpoint may be reused.
    """
    diffs = []
    for key in sorted(set(before) | set(after)):
        if key in _VOLATILE or key == "clock_fraction_of_max":
            continue
        old, new = before.get(key), after.get(key)
        if old != new:
            diffs.append(f"{key}: {old} -> {new}")
    return diffs


def format_gpu_state(state: dict) -> str:
    if not state.get("available"):
        return "gpu: unavailable"
    return (
        f"gpu: {state['name']} driver {state['driver_version']}  "
        f"{state['pstate']} {state['clocks_sm_mhz']}/{state['clocks_max_sm_mhz']} MHz  "
        f"{state['enforced_power_limit_w']}W limit (default {state['default_power_limit_w']}W)"
    )
