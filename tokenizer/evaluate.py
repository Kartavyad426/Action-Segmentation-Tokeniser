"""Tokenizer-only evaluation: reconstruction error, codebook utilization, boundary alignment."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from tokenizer.windowing import demo_to_sequence


@torch.no_grad()
def reconstruction_error(model, dataset, device: str = "cpu") -> float:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=64)
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        total += out["recon_loss"].item() * batch.shape[0]
        count += batch.shape[0]
    return total / count


@torch.no_grad()
def codebook_utilization(model, dataset, num_codes: int, device: str = "cpu") -> float:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=64)
    counts = torch.zeros(num_codes)
    for batch in loader:
        batch = batch.to(device)
        _, indices = model.encode_tokens(batch)
        counts += torch.bincount(indices.cpu(), minlength=num_codes).float()

    probs = counts / counts.sum().clamp(min=1)
    nonzero = probs[probs > 0]
    entropy = -(nonzero * nonzero.log()).sum().item()
    max_entropy = np.log(num_codes)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def boundary_alignment_report(
    indices: np.ndarray, centers: np.ndarray, low_level_segments, tolerance_s: float = 0.2
) -> dict:
    """How well token changes line up with ground-truth action boundaries.

    Recall alone is not a usable gate: a tokenizer that emits a different code every window
    hits every boundary by accident and scores a perfect 1.0 while carrying no information.
    So this reports three things recall cannot express on its own:

    - `change_precision` -- of the token changes the tokenizer actually made, how many landed
      near a real boundary. This is what punishes changing constantly.
    - `chance_recall` -- the recall the same *number* of changes would achieve if they were
      scattered uniformly at random, so `lift_over_chance` says whether the placement carries
      information or the score is just a function of how often the tokenizer flips.
    - `segment_purity` -- within a ground-truth segment, the fraction of tokens holding that
      segment's most common code. A stable tokenizer scores 1.0; a flickering one scores near
      1/n_codes.

    `f1` is the harmonic mean of recall and precision, and is the number worth gating on.
    """
    changed = np.asarray(indices)[1:] != np.asarray(indices)[:-1]
    change_times = np.asarray(centers)[1:][changed]
    boundary_times = sorted(
        {start for start, _, _ in low_level_segments} | {end for _, end, _ in low_level_segments}
    )

    empty = {
        "boundary_recall": 0.0, "change_precision": 0.0, "f1": 0.0,
        "chance_recall": 0.0, "lift_over_chance": 0.0,
        "segment_purity": _segment_purity(indices, centers, low_level_segments),
        "n_changes": int(len(change_times)), "change_rate": float(changed.mean()) if len(changed) else 0.0,
    }
    if len(change_times) == 0 or not boundary_times:
        return empty

    boundary_times = np.asarray(boundary_times, dtype=float)
    hits = np.array([np.min(np.abs(change_times - b)) <= tolerance_s for b in boundary_times])
    recall = float(hits.mean())

    near = np.array([np.min(np.abs(boundary_times - t)) <= tolerance_s for t in change_times])
    precision = float(near.mean())

    # If the same number of changes were scattered uniformly over the demo, each boundary has
    # a +/-tolerance window to be hit; edge effects at the very start/end are ignored.
    duration = float(centers[-1] - centers[0]) if len(centers) > 1 else 0.0
    n = len(change_times)
    if duration > 0:
        p_miss_one = max(0.0, 1.0 - min(2 * tolerance_s, duration) / duration)
        chance = 1.0 - p_miss_one ** n
    else:
        chance = 0.0

    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    return {
        "boundary_recall": recall,
        "change_precision": precision,
        "f1": f1,
        "chance_recall": chance,
        "lift_over_chance": recall - chance,
        "segment_purity": _segment_purity(indices, centers, low_level_segments),
        "n_changes": int(n),
        "change_rate": float(changed.mean()),
    }


def _segment_purity(indices, centers, low_level_segments) -> float:
    indices, centers = np.asarray(indices), np.asarray(centers)
    purities = []
    for start, end, _ in low_level_segments:
        inside = indices[(centers >= start) & (centers < end)]
        if len(inside) == 0:
            continue
        purities.append(np.bincount(inside).max() / len(inside))
    return float(np.mean(purities)) if purities else 0.0


@torch.no_grad()
def boundary_alignment_score(
    model, telemetry, grid, low_level_segments, window: int = 15, tolerance_s: float = 0.2, device: str = "cpu"
) -> dict:
    """Runs the tokenizer over one demo and reports boundary alignment. See
    `boundary_alignment_report` for what the returned fields mean."""
    model = model.to(device).eval()
    windows, centers = demo_to_sequence(telemetry, grid, window=window)
    x = torch.from_numpy(windows.astype(np.float32)).to(device)
    _, indices = model.encode_tokens(x)
    return boundary_alignment_report(indices.cpu().numpy(), centers, low_level_segments, tolerance_s)
