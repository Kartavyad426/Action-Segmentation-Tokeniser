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


@torch.no_grad()
def boundary_alignment_score(
    model, telemetry, grid, low_level_segments, window: int = 15, tolerance_s: float = 0.2, device: str = "cpu"
) -> float:
    model = model.to(device).eval()
    windows, centers = demo_to_sequence(telemetry, grid, window=window)
    x = torch.from_numpy(windows.astype(np.float32)).to(device)
    _, indices = model.encode_tokens(x)
    indices = indices.cpu().numpy()

    change_times = centers[1:][indices[1:] != indices[:-1]]
    if len(change_times) == 0:
        return 0.0

    boundary_times = sorted({start for start, _, _ in low_level_segments} | {end for _, end, _ in low_level_segments})
    if not boundary_times:
        return 0.0

    hits = 0
    for boundary in boundary_times:
        if np.min(np.abs(change_times - boundary)) <= tolerance_s:
            hits += 1
    return hits / len(boundary_times)
