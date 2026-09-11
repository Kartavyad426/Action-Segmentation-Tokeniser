"""Aggregate F1@50 across repeated runs, to separate a real effect from run-to-run noise.

    python -m temporal_classifier.summarize runs/
"""

import json
import os
import statistics


def collect_runs(runs_base: str) -> list[dict]:
    """Every run directory that actually produced results, newest last."""
    runs = []
    for name in sorted(os.listdir(runs_base)):
        path = os.path.join(runs_base, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        results_path = os.path.join(path, "results.json")
        if not os.path.exists(results_path):
            continue  # aborted before any arm finished
        try:
            results = json.load(open(results_path))
            config = json.load(open(os.path.join(path, "config.json")))
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            continue
        runs.append({"dir": path, "name": name, "results": results, "config": config})
    return runs


def summarize(runs: list[dict], control: str = "telemetry_only") -> dict:
    """Per-arm spread, plus the delta against the control computed *within* each run.

    Differencing two arm means would discard the pairing: if runs vary a lot but vision
    helps consistently within each run, the paired delta shows that and the difference of
    means hides it in the noise.
    """
    by_arm = {}
    for run in runs:
        for arm, result in run["results"].get("arms", {}).items():
            by_arm.setdefault(arm, []).append((run["name"], result["val_f1"]))

    summary = {}
    for arm, entries in by_arm.items():
        scores = [s for _, s in entries]
        deltas = []
        for run in runs:
            arms = run["results"].get("arms", {})
            if arm in arms and control in arms and arm != control:
                deltas.append(arms[arm]["val_f1"] - arms[control]["val_f1"])
        summary[arm] = {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "runs": entries,
            "delta_mean": statistics.fmean(deltas) if deltas else None,
            "delta_std": statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None),
        }
    return summary


def format_summary(summary: dict, control: str = "telemetry_only") -> str:
    order = [control] + sorted(a for a in summary if a != control)
    lines = [
        "F1@50 across repeated runs",
        "-" * 74,
        f"{'arm':<24} {'n':>2} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'delta':>10}",
    ]
    for arm in order:
        if arm not in summary:
            continue
        s = summary[arm]
        delta = "--" if s["delta_mean"] is None else f"{s['delta_mean']:+.4f}"
        if s["delta_mean"] is not None and s["delta_std"]:
            delta += f" ±{s['delta_std']:.4f}"
        lines.append(
            f"{arm:<24} {s['n']:>2} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['min']:>8.4f} {s['max']:>8.4f} {delta:>10}"
        )
    lines.append("")
    lines.append("delta is computed within each run against the control arm, then averaged.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_base", nargs="?", default="runs")
    args = parser.parse_args()

    found = collect_runs(args.runs_base)
    if not found:
        print(f"no runs with results in {args.runs_base}")
    else:
        print(format_summary(summarize(found)))
        print(f"\n{len(found)} run(s): " + ", ".join(r["name"] for r in found))
