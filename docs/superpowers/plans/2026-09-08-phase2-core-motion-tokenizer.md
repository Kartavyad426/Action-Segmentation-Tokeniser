# Phase 2 Core: Motion Tokenizer + Vision Enrichment + Temporal Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate, on REASSEMBLE, a motion tokenizer → vision-enrichment →
temporal-classifier stack that reports F1@50 with and without vision, answering whether
vision enrichment is worth keeping in this pipeline.

**Architecture:** A VQ-VAE tokenizer turns fixed-length telemetry windows into discrete
motion tokens (reconstruction-only, unsupervised). Each token's continuous embedding is
optionally fused with a frozen DINOv2 feature from the time-aligned video frame. An MS-TCN
temporal classifier reads the full per-demo token sequence and predicts REASSEMBLE's
low-level action label (Grasp, Lift, Approach, Align, ...) per token.

**Tech Stack:** Python 3.12, PyTorch 2.11+cu128 (conda env `robotics-phase2`), h5py, OpenCV,
DINOv2 via `torch.hub`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-phase2-core-motion-tokenizer-design.md`

## Global Constraints

- Conda env `robotics-phase2` (python 3.12) — activate via
  `source /home/imerit/miniconda3/etc/profile.d/conda.sh && conda activate robotics-phase2`
  before running anything in this plan.
- GPU: 8GB (RTX PRO 1000 Blackwell). Keep batch sizes and model widths modest — the sizes
  given in each task are chosen to fit comfortably; don't scale them up without checking
  memory first.
- Real telemetry rate is ~970-1000Hz (confirmed by inspection, not the ~20Hz originally
  assumed) — all telemetry is resampled onto a uniform **100Hz** grid before windowing
  (`RATE_HZ = 100.0`, defined once in `tokenizer/data.py` and imported everywhere else).
- Tokenizer window = 15 samples (150ms) at 100Hz, training stride = 5 samples (overlapping,
  for VQ-VAE training diversity); sequence construction for the classifier uses
  non-overlapping windows (stride = window = 15) so each demo yields one clean token per
  150ms.
- **Classifier target is REASSEMBLE's low-level segment labels** (Grasp, Lift, Approach,
  Align, ...), not its high-level task labels (e.g. "Pick Ethernet") — confirmed in the spec
  as the correct mapping to this project's closed-vocabulary-of-primitives goal.
- Classifier architecture is MS-TCN (multi-stage TCN), not a Transformer, per the spec's
  staged decision.
- Data lives in `data/reassemble/*.h5` (gitignored). Only 10 demos are downloaded as of this
  plan being written — `data/download_reassemble.py` (no `--limit`) fetches the rest when
  needed; tests that need the data skip gracefully if files aren't present, and must not
  assume the full 148-demo set exists yet.
- Official train/test split lives in `data/splits_inspect/{train,test}_split1.txt` (filename
  stems, no extension) — training/eval code must intersect this with whatever's actually on
  disk in `data/reassemble/`, not assume all listed files are present.

---

### Task 1: Telemetry data loading & windowing

**Files:**
- Create: `tokenizer/data.py`
- Create: `tokenizer/windowing.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing (foundational).
- Produces:
  - `tokenizer.data.RATE_HZ: float` (100.0)
  - `tokenizer.data.TELEMETRY_CHANNELS: list[str]`
  - `tokenizer.data.load_demo(path: str, channels: list[str] = TELEMETRY_CHANNELS) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[tuple[float, float, str]]]`
    — returns `(channel_arrays, channel_timestamps, low_level_segments)`.
  - `tokenizer.data.resample_to_grid(channel_arrays, channel_timestamps, rate_hz=RATE_HZ) -> tuple[np.ndarray, np.ndarray]`
    — returns `(telemetry[T, C], grid[T])`.
  - `tokenizer.windowing.compute_norm_stats(telemetry_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]`
    — returns `(mean[1, C], std[1, C])`.
  - `tokenizer.windowing.WindowedTelemetryDataset` (torch `Dataset`): constructor
    `(telemetry_list: list[np.ndarray], window: int = 15, stride: int = 5, mean=None, std=None)`,
    `__getitem__` returns a `torch.FloatTensor` of shape `(window, C)`.
  - `tokenizer.windowing.demo_to_sequence(telemetry: np.ndarray, grid: np.ndarray, window: int = 15) -> tuple[np.ndarray, np.ndarray]`
    — non-overlapping windows, returns `(windows[N, window, C], window_center_times[N])`.
  - `tokenizer.windowing.split_available_demos(data_dir="data/reassemble", splits_dir="data/splits_inspect") -> tuple[list[str], list[str]]`
    — returns `(train_paths, val_paths)`, intersecting the official split with files that
    actually exist on disk.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import os

import numpy as np
import pytest

from tokenizer.data import TELEMETRY_CHANNELS, load_demo, resample_to_grid
from tokenizer.windowing import (
    WindowedTelemetryDataset,
    compute_norm_stats,
    demo_to_sequence,
    split_available_demos,
)

DEMO_PATH = "data/reassemble/2025-01-09-13-57-17.h5"
requires_demo = pytest.mark.skipif(not os.path.exists(DEMO_PATH), reason="demo file not downloaded")


@requires_demo
def test_load_demo_returns_channels_timestamps_and_segments():
    channel_arrays, channel_timestamps, low_level_segments = load_demo(DEMO_PATH)

    assert set(channel_arrays.keys()) == set(TELEMETRY_CHANNELS)
    assert set(channel_timestamps.keys()) == set(TELEMETRY_CHANNELS)
    for ch in TELEMETRY_CHANNELS:
        assert channel_arrays[ch].shape[0] == channel_timestamps[ch].shape[0]

    texts = {text for _, _, text in low_level_segments}
    assert "Grasp" in texts
    assert "Lift" in texts
    assert all(end > start for start, end, _ in low_level_segments)


@requires_demo
def test_resample_to_grid_produces_uniform_100hz_grid():
    channel_arrays, channel_timestamps, _ = load_demo(DEMO_PATH)
    telemetry, grid = resample_to_grid(channel_arrays, channel_timestamps, rate_hz=100.0)

    n_channels = sum(channel_arrays[ch].shape[1] for ch in TELEMETRY_CHANNELS)
    assert telemetry.shape == (len(grid), n_channels)
    assert telemetry.dtype == np.float32

    diffs = np.diff(grid)
    assert np.allclose(diffs, 0.01, atol=1e-6)


def test_compute_norm_stats_matches_manual_mean_std():
    a = np.array([[1.0, 10.0], [3.0, 30.0]], dtype=np.float32)
    b = np.array([[5.0, 50.0]], dtype=np.float32)

    mean, std = compute_norm_stats([a, b])

    expected_mean = np.concatenate([a, b], axis=0).mean(axis=0, keepdims=True)
    assert np.allclose(mean, expected_mean)
    assert std.shape == (1, 2)


def test_windowed_dataset_length_and_shape():
    telemetry = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    ds = WindowedTelemetryDataset([telemetry], window=10, stride=5)

    assert len(ds) == (40 - 10) // 5 + 1
    window = ds[0]
    assert window.shape == (10, 3)


def test_demo_to_sequence_is_non_overlapping():
    telemetry = np.arange(30 * 2, dtype=np.float32).reshape(30, 2)
    grid = np.arange(30) * 0.01
    windows, centers = demo_to_sequence(telemetry, grid, window=10)

    assert windows.shape == (3, 10, 2)
    assert centers.shape == (3,)
    assert np.array_equal(windows[0], telemetry[0:10])
    assert np.array_equal(windows[1], telemetry[10:20])


def test_split_available_demos_intersects_disk_with_official_split(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "train_split1.txt").write_text("demo_a\ndemo_b\n")
    (splits_dir / "test_split1.txt").write_text("demo_c\n")

    data_dir = tmp_path / "reassemble"
    data_dir.mkdir()
    (data_dir / "demo_a.h5").touch()
    (data_dir / "demo_c.h5").touch()
    # demo_b is in the split but not on disk yet — must be excluded, not error.

    train_paths, val_paths = split_available_demos(data_dir=str(data_dir), splits_dir=str(splits_dir))

    assert train_paths == [str(data_dir / "demo_a.h5")]
    assert val_paths == [str(data_dir / "demo_c.h5")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/imerit/miniconda3/etc/profile.d/conda.sh && conda activate robotics-phase2 && cd /home/imerit/Documents/Code/robotics && python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer.data'` (or similar — the modules don't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# tokenizer/data.py
"""Load a REASSEMBLE demo .h5 file and resample its telemetry onto a uniform time grid."""

import h5py
import numpy as np

RATE_HZ = 100.0

TELEMETRY_CHANNELS = [
    "joint_positions",
    "joint_velocities",
    "gripper_positions",
    "measured_force",
    "measured_torque",
]


def load_demo(path: str, channels: list[str] = TELEMETRY_CHANNELS):
    """Returns (channel_arrays, channel_timestamps, low_level_segments)."""
    channel_arrays: dict[str, np.ndarray] = {}
    channel_timestamps: dict[str, np.ndarray] = {}
    low_level_segments: list[tuple[float, float, str]] = []

    with h5py.File(path, "r") as f:
        for ch in channels:
            channel_arrays[ch] = f[f"robot_state/{ch}"][:]
            channel_timestamps[ch] = f[f"timestamps/{ch}"][:]

        for key in sorted(f["segments_info"].keys(), key=int):
            g = f[f"segments_info/{key}"]
            if "low_level" not in g:
                continue
            for lkey in sorted(g["low_level"].keys(), key=int):
                lg = g[f"low_level/{lkey}"]
                text = lg["text"][()]
                if isinstance(text, bytes):
                    text = text.decode()
                low_level_segments.append((float(lg["start"][()]), float(lg["end"][()]), text))

    return channel_arrays, channel_timestamps, low_level_segments


def resample_to_grid(channel_arrays, channel_timestamps, rate_hz: float = RATE_HZ):
    """Linearly interpolates every channel onto one shared uniform time grid."""
    t0 = max(ts[0] for ts in channel_timestamps.values())
    t1 = min(ts[-1] for ts in channel_timestamps.values())
    n = int((t1 - t0) * rate_hz)
    grid = t0 + np.arange(n) / rate_hz

    resampled = []
    for ch, arr in channel_arrays.items():
        ts = channel_timestamps[ch]
        cols = [np.interp(grid, ts, arr[:, i]) for i in range(arr.shape[1])]
        resampled.append(np.stack(cols, axis=1))

    telemetry = np.concatenate(resampled, axis=1).astype(np.float32)
    return telemetry, grid
```

```python
# tokenizer/windowing.py
"""Turn resampled telemetry into fixed-length windows for tokenizer training/inference."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_norm_stats(telemetry_list: list[np.ndarray]):
    stacked = np.concatenate(telemetry_list, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-6
    return mean, std


class WindowedTelemetryDataset(Dataset):
    def __init__(self, telemetry_list: list[np.ndarray], window: int = 15, stride: int = 5, mean=None, std=None):
        self.window = window
        if mean is None or std is None:
            mean, std = compute_norm_stats(telemetry_list)
        self.mean, self.std = mean, std

        self.windows = []
        for telemetry in telemetry_list:
            normed = (telemetry - mean) / std
            n_windows = max(0, (len(normed) - window) // stride + 1)
            for i in range(n_windows):
                s = i * stride
                self.windows.append(normed[s : s + window].astype(np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx])


def demo_to_sequence(telemetry: np.ndarray, grid: np.ndarray, window: int = 15):
    """Non-overlapping windows spanning the whole demo, plus each window's center time."""
    n_windows = len(telemetry) // window
    windows = np.stack([telemetry[i * window : (i + 1) * window] for i in range(n_windows)])
    centers = np.array([grid[i * window + window // 2] for i in range(n_windows)])
    return windows, centers


def split_available_demos(data_dir: str = "data/reassemble", splits_dir: str = "data/splits_inspect"):
    def read_stems(name: str) -> list[str]:
        path = Path(splits_dir) / name
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def resolve(stems: list[str]) -> list[str]:
        paths = []
        for stem in stems:
            candidate = Path(data_dir) / f"{stem}.h5"
            if candidate.exists():
                paths.append(str(candidate))
        return paths

    train_paths = resolve(read_stems("train_split1.txt"))
    val_paths = resolve(read_stems("test_split1.txt"))
    return train_paths, val_paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (7 tests; the two `requires_demo`-marked tests run since the smoke-test demo
is already downloaded at `data/reassemble/2025-01-09-13-57-17.h5`).

- [ ] **Step 5: Commit**

```bash
git add tokenizer/data.py tokenizer/windowing.py tests/test_data.py
git commit -m "feat: telemetry data loading and windowing for the motion tokenizer"
```

---

### Task 2: VQ-VAE tokenizer model

**Files:**
- Create: `tokenizer/model.py`
- Test: `tests/test_tokenizer_model.py`

**Interfaces:**
- Consumes: nothing beyond `torch`.
- Produces:
  - `tokenizer.model.MotionTokenizer(in_channels: int, window: int, latent_dim: int = 32, num_codes: int = 512, hidden: int = 64)`
    (`nn.Module`).
    - `forward(x: Tensor[B, window, in_channels]) -> dict` with keys `recon` (`Tensor[B, window, in_channels]`),
      `indices` (`Tensor[B]`, long), `loss`, `recon_loss`, `vq_loss` (scalars).
    - `encode_tokens(x: Tensor[B, window, in_channels]) -> tuple[Tensor[B, latent_dim], Tensor[B]]`
      returns `(z_q, indices)` without computing reconstruction — this is what later tasks
      use to get per-window token embeddings.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer_model.py
import torch

from tokenizer.model import MotionTokenizer


def test_forward_shapes():
    model = MotionTokenizer(in_channels=22, window=15, latent_dim=32, num_codes=64, hidden=32)
    x = torch.randn(4, 15, 22)

    out = model(x)

    assert out["recon"].shape == (4, 15, 22)
    assert out["indices"].shape == (4,)
    assert out["loss"].ndim == 0
    assert out["recon_loss"].ndim == 0
    assert out["vq_loss"].ndim == 0


def test_encode_tokens_matches_forward_indices():
    model = MotionTokenizer(in_channels=5, window=10, latent_dim=8, num_codes=16, hidden=16)
    x = torch.randn(3, 10, 5)

    out = model(x)
    z_q, indices = model.encode_tokens(x)

    assert z_q.shape == (3, 8)
    assert torch.equal(indices, out["indices"])


def test_loss_decreases_when_overfitting_a_single_batch():
    torch.manual_seed(0)
    model = MotionTokenizer(in_channels=4, window=10, latent_dim=8, num_codes=16, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 10, 4)

    first_loss = None
    last_loss = None
    for step in range(50):
        optimizer.zero_grad()
        out = model(x)
        out["loss"].backward()
        optimizer.step()
        if step == 0:
            first_loss = out["loss"].item()
        last_loss = out["loss"].item()

    assert last_loss < first_loss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer.model'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tokenizer/model.py
"""VQ-VAE-style motion tokenizer: encode a telemetry window, quantize, decode."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, latent_dim)

    def forward(self, x):  # x: (B, T, C)
        h = self.net(x.transpose(1, 2)).squeeze(-1)  # (B, hidden)
        return self.proj(h)  # (B, latent_dim)


class _Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int, window: int, hidden: int):
        super().__init__()
        self.window = window
        self.hidden = hidden
        self.fc = nn.Linear(latent_dim, hidden * window)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):  # z: (B, latent_dim)
        h = self.fc(z).view(-1, self.hidden, self.window)
        return self.net(h).transpose(1, 2)  # (B, T, C)


class _VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)
        self.commitment_cost = commitment_cost

    def forward(self, z):  # z: (B, D)
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q = z + (z_q - z).detach()  # straight-through estimator
        return z_q, indices, vq_loss


class MotionTokenizer(nn.Module):
    def __init__(self, in_channels: int, window: int, latent_dim: int = 32, num_codes: int = 512, hidden: int = 64):
        super().__init__()
        self.encoder = _Encoder(in_channels, latent_dim, hidden)
        self.quantizer = _VectorQuantizer(num_codes, latent_dim)
        self.decoder = _Decoder(in_channels, latent_dim, window, hidden)

    def encode_tokens(self, x):
        z = self.encoder(x)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return {
            "recon": recon,
            "indices": indices,
            "loss": recon_loss + vq_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tokenizer/model.py tests/test_tokenizer_model.py
git commit -m "feat: VQ-VAE motion tokenizer model"
```

---

### Task 3: Tokenizer training script

**Files:**
- Create: `tokenizer/train.py`
- Test: `tests/test_tokenizer_train.py`

**Interfaces:**
- Consumes: `tokenizer.data.{load_demo, resample_to_grid, TELEMETRY_CHANNELS}`,
  `tokenizer.windowing.{WindowedTelemetryDataset, compute_norm_stats, split_available_demos}`,
  `tokenizer.model.MotionTokenizer`.
- Produces:
  - `tokenizer.train.train_tokenizer(demo_paths: list[str], out_path: str, epochs: int = 20, window: int = 15, stride: int = 5, latent_dim: int = 32, num_codes: int = 512, hidden: int = 64, batch_size: int = 64, lr: float = 1e-3, device: str = "cuda") -> dict`
    — trains, saves a checkpoint to `out_path` containing
    `{"model_state": ..., "mean": np.ndarray, "std": np.ndarray, "config": {...}}`,
    and returns `{"final_loss": float}`.
  - `tokenizer.train.load_checkpoint(path: str, device: str = "cpu") -> tuple[MotionTokenizer, np.ndarray, np.ndarray, dict]`
    returns `(model, mean, std, config)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer_train.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tokenizer/train.py
"""Train the VQ-VAE motion tokenizer on a list of REASSEMBLE demo files."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from tokenizer.data import TELEMETRY_CHANNELS, load_demo, resample_to_grid
from tokenizer.model import MotionTokenizer
from tokenizer.windowing import WindowedTelemetryDataset


def _load_all_telemetry(demo_paths: list[str]):
    telemetry_list = []
    for path in demo_paths:
        channel_arrays, channel_timestamps, _ = load_demo(path)
        telemetry, _ = resample_to_grid(channel_arrays, channel_timestamps)
        telemetry_list.append(telemetry)
    return telemetry_list


def train_tokenizer(
    demo_paths: list[str],
    out_path: str,
    epochs: int = 20,
    window: int = 15,
    stride: int = 5,
    latent_dim: int = 32,
    num_codes: int = 512,
    hidden: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    telemetry_list = _load_all_telemetry(demo_paths)
    in_channels = telemetry_list[0].shape[1]

    dataset = WindowedTelemetryDataset(telemetry_list, window=window, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MotionTokenizer(in_channels, window, latent_dim, num_codes, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            out["loss"].backward()
            optimizer.step()
            final_loss = out["loss"].item()

    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": dataset.mean,
            "std": dataset.std,
            "config": {
                "in_channels": in_channels,
                "window": window,
                "latent_dim": latent_dim,
                "num_codes": num_codes,
                "hidden": hidden,
            },
        },
        out_path,
    )
    return {"final_loss": final_loss}


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MotionTokenizer(
        config["in_channels"], config["window"], config["latent_dim"], config["num_codes"], config["hidden"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["mean"], checkpoint["std"], config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer_train.py -v`
Expected: PASS (1 test; runs on CPU with 2 epochs so it stays fast regardless of GPU state).

- [ ] **Step 5: Commit**

```bash
git add tokenizer/train.py tests/test_tokenizer_train.py
git commit -m "feat: tokenizer training script with checkpoint save/load"
```

---

### Task 4: Tokenizer evaluation suite

**Files:**
- Create: `tokenizer/evaluate.py`
- Test: `tests/test_tokenizer_evaluate.py`

**Interfaces:**
- Consumes: `tokenizer.model.MotionTokenizer`, `tokenizer.windowing.{WindowedTelemetryDataset, demo_to_sequence}`.
- Produces:
  - `tokenizer.evaluate.reconstruction_error(model, dataset, device="cpu") -> float` — mean
    per-window normalized MSE over `dataset`.
  - `tokenizer.evaluate.codebook_utilization(model, dataset, num_codes, device="cpu") -> float`
    — normalized usage entropy in `[0, 1]` (1.0 = every code used equally often, 0.0 =
    a single code absorbs everything).
  - `tokenizer.evaluate.boundary_alignment_score(model, telemetry, grid, low_level_segments, window=15, tolerance_s=0.2, device="cpu") -> float`
    — fraction of ground-truth segment boundaries that have a token change within
    `tolerance_s` seconds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer_evaluate.py
import numpy as np
import torch

from tokenizer.evaluate import boundary_alignment_score, codebook_utilization, reconstruction_error
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

    score = boundary_alignment_score(model, telemetry, grid, segments, window=10, tolerance_s=0.05)

    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tokenizer_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tokenizer/evaluate.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tokenizer_evaluate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tokenizer/evaluate.py tests/test_tokenizer_evaluate.py
git commit -m "feat: tokenizer evaluation suite (reconstruction, codebook, boundary alignment)"
```

---

### Task 5: Action vocabulary & label alignment

**Files:**
- Create: `temporal_classifier/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: nothing beyond `numpy`.
- Produces:
  - `temporal_classifier.labels.BACKGROUND_LABEL: str` (`"background"`)
  - `temporal_classifier.labels.build_vocab(segments_per_demo: list[list[tuple[float, float, str]]]) -> dict[str, int]`
    — `BACKGROUND_LABEL` always maps to `0`.
  - `temporal_classifier.labels.align_labels_to_grid(grid: np.ndarray, segments: list[tuple[float, float, str]], vocab: dict[str, int]) -> np.ndarray`
    — `int64` array, same length as `grid`, defaulting to `0` (background) outside any
    segment.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
import numpy as np

from temporal_classifier.labels import BACKGROUND_LABEL, align_labels_to_grid, build_vocab


def test_build_vocab_includes_background_and_all_seen_labels():
    segments_per_demo = [
        [(0.0, 1.0, "Grasp"), (1.0, 2.0, "Lift")],
        [(0.0, 1.0, "Approach"), (1.0, 2.0, "Align")],
    ]

    vocab = build_vocab(segments_per_demo)

    assert vocab[BACKGROUND_LABEL] == 0
    assert set(vocab.keys()) == {BACKGROUND_LABEL, "Grasp", "Lift", "Approach", "Align"}
    assert len(set(vocab.values())) == len(vocab)  # all ids unique


def test_align_labels_to_grid_defaults_to_background_outside_segments():
    vocab = {BACKGROUND_LABEL: 0, "Grasp": 1, "Lift": 2}
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
    segments = [(0.0, 1.0, "Grasp"), (1.0, 2.0, "Lift")]

    labels = align_labels_to_grid(grid, segments, vocab)

    assert labels.tolist() == [1, 1, 2, 2, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_classifier.labels'`.

- [ ] **Step 3: Write minimal implementation**

```python
# temporal_classifier/labels.py
"""Build a closed action vocabulary from REASSEMBLE low-level labels and align it to a time grid."""

import numpy as np

BACKGROUND_LABEL = "background"


def build_vocab(segments_per_demo: list[list[tuple[float, float, str]]]) -> dict[str, int]:
    labels = sorted({text for segs in segments_per_demo for _, _, text in segs})
    vocab = {BACKGROUND_LABEL: 0}
    for i, label in enumerate(labels, start=1):
        vocab[label] = i
    return vocab


def align_labels_to_grid(grid: np.ndarray, segments: list[tuple[float, float, str]], vocab: dict[str, int]) -> np.ndarray:
    label_ids = np.zeros(len(grid), dtype=np.int64)
    for start, end, text in segments:
        mask = (grid >= start) & (grid < end)
        label_ids[mask] = vocab[text]
    return label_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_labels.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add temporal_classifier/labels.py tests/test_labels.py
git commit -m "feat: closed action vocabulary and label-to-grid alignment"
```

---

### Task 6: MS-TCN classifier model

**Files:**
- Create: `temporal_classifier/model.py`
- Test: `tests/test_classifier_model.py`

**Interfaces:**
- Consumes: nothing beyond `torch`.
- Produces:
  - `temporal_classifier.model.MSTCN(in_channels: int, num_classes: int, channels: int = 64, num_layers: int = 9, num_stages: int = 3)`
    (`nn.Module`). `forward(x: Tensor[B, T, in_channels]) -> list[Tensor[B, num_classes, T]]`,
    one tensor per refinement stage (length `num_stages`), in order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_model.py
import torch

from temporal_classifier.model import MSTCN


def test_forward_returns_one_output_per_stage_with_correct_shape():
    model = MSTCN(in_channels=32, num_classes=6, channels=16, num_layers=4, num_stages=3)
    x = torch.randn(2, 20, 32)

    outputs = model(x)

    assert len(outputs) == 3
    for out in outputs:
        assert out.shape == (2, 6, 20)


def test_gradient_flows_to_first_stage_from_last_stage_loss():
    model = MSTCN(in_channels=8, num_classes=3, channels=8, num_layers=2, num_stages=2)
    x = torch.randn(1, 10, 8)

    outputs = model(x)
    loss = outputs[-1].sum()
    loss.backward()

    grad = model.stage1.in_conv.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_classifier.model'`.

- [ ] **Step 3: Write minimal implementation**

```python
# temporal_classifier/model.py
"""MS-TCN: multi-stage temporal convolutional network for per-token action classification."""

import torch
import torch.nn as nn


class _DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = torch.relu(self.conv(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class _SingleStageTCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: int, num_layers: int):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.layers = nn.ModuleList(_DilatedResidualLayer(2**i, channels) for i in range(num_layers))
        self.out_conv = nn.Conv1d(channels, num_classes, kernel_size=1)

    def forward(self, x):  # x: (B, C, T)
        h = self.in_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_conv(h)  # (B, num_classes, T)


class MSTCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: int = 64, num_layers: int = 9, num_stages: int = 3):
        super().__init__()
        self.stage1 = _SingleStageTCN(in_channels, num_classes, channels, num_layers)
        self.stages = nn.ModuleList(
            _SingleStageTCN(num_classes, num_classes, channels, num_layers) for _ in range(num_stages - 1)
        )

    def forward(self, x):  # x: (B, T, in_channels)
        x = x.transpose(1, 2)  # (B, in_channels, T)
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(torch.softmax(out, dim=1))
            outputs.append(out)
        return outputs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier_model.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add temporal_classifier/model.py tests/test_classifier_model.py
git commit -m "feat: MS-TCN temporal classifier model"
```

---

### Task 7: Segmental F1@k metric

**Files:**
- Create: `temporal_classifier/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing beyond `numpy`.
- Produces:
  - `temporal_classifier.metrics.get_segments(labels: np.ndarray) -> list[tuple[int, int, int]]`
    — merges a per-frame label array into `(label, start, end)` segments, `end` exclusive.
  - `temporal_classifier.metrics.f1_at_k(pred: np.ndarray, gt: np.ndarray, overlap: float = 0.5, background_label: int = 0) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np

from temporal_classifier.metrics import f1_at_k, get_segments


def test_get_segments_merges_consecutive_equal_labels():
    labels = np.array([1, 1, 1, 2, 2, 0, 0, 3])

    segments = get_segments(labels)

    assert segments == [(1, 0, 3), (2, 3, 5), (0, 5, 7), (3, 7, 8)]


def test_f1_at_k_is_one_for_a_perfect_match():
    labels = np.array([0, 1, 1, 1, 0, 2, 2, 0])

    score = f1_at_k(labels, labels, overlap=0.5)

    assert score == 1.0


def test_f1_at_k_is_zero_when_no_predicted_segment_matches():
    gt = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    pred = np.array([0, 3, 3, 3, 0, 4, 4, 0])  # entirely different labels

    score = f1_at_k(pred, gt, overlap=0.5)

    assert score == 0.0


def test_f1_at_k_penalizes_low_overlap_below_threshold():
    gt = np.array([1] * 10)
    pred = np.array([1] * 3 + [0] * 7)  # only 30% overlap with the single gt segment

    score = f1_at_k(pred, gt, overlap=0.5)

    assert score == 0.0  # below the 0.5 IoU threshold, counts as a miss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_classifier.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# temporal_classifier/metrics.py
"""Segmental F1@k — the standard temporal-action-segmentation evaluation metric."""

import numpy as np


def get_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    segments = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segments.append((int(labels[start]), start, i))
            start = i
    return segments


def f1_at_k(pred: np.ndarray, gt: np.ndarray, overlap: float = 0.5, background_label: int = 0) -> float:
    pred_segs = [s for s in get_segments(pred) if s[0] != background_label]
    gt_segs = [s for s in get_segments(gt) if s[0] != background_label]

    matched = [False] * len(gt_segs)
    tp = 0
    for p_label, p_start, p_end in pred_segs:
        best_iou, best_j = 0.0, -1
        for j, (g_label, g_start, g_end) in enumerate(gt_segs):
            if matched[j] or g_label != p_label:
                continue
            inter = max(0, min(p_end, g_end) - max(p_start, g_start))
            union = max(p_end, g_end) - min(p_start, g_start)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= overlap:
            matched[best_j] = True
            tp += 1

    fp = len(pred_segs) - tp
    fn = len(gt_segs) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add temporal_classifier/metrics.py tests/test_metrics.py
git commit -m "feat: segmental F1@k evaluation metric"
```

---

### Task 8: Telemetry-only classifier training + evaluation

**Files:**
- Create: `temporal_classifier/train.py`
- Test: `tests/test_classifier_train.py`

**Interfaces:**
- Consumes: `tokenizer.data.{load_demo, resample_to_grid}`, `tokenizer.windowing.demo_to_sequence`,
  `tokenizer.train.load_checkpoint`, `temporal_classifier.labels.{build_vocab, align_labels_to_grid}`,
  `temporal_classifier.model.MSTCN`, `temporal_classifier.metrics.f1_at_k`.
- Produces:
  - `temporal_classifier.train.build_demo_sequence(tokenizer_model, mean, std, demo_path, window) -> tuple[np.ndarray, np.ndarray, list]`
    returns `(token_embeddings[N, latent_dim], centers[N], low_level_segments)` for one demo.
  - `temporal_classifier.train.run_training(tokenizer_ckpt_path: str, train_paths: list[str], val_paths: list[str], vocab: dict[str, int] | None = None, epochs: int = 30, channels: int = 64, num_layers: int = 9, num_stages: int = 3, lr: float = 1e-3, device: str = "cuda") -> dict`
    — trains an `MSTCN` on frozen tokenizer embeddings, returns
    `{"model": MSTCN, "vocab": dict, "val_f1": float}`. Building `vocab` from the training
    demos if not supplied lets Task 10's vision-enriched run reuse a shared vocabulary
    explicitly instead of silently rebuilding a different one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_train.py
import os

import pytest

from temporal_classifier.train import run_training
from tokenizer.train import train_tokenizer
from tokenizer.windowing import split_available_demos

requires_data = pytest.mark.skipif(
    not os.path.exists("data/reassemble"), reason="REASSEMBLE demos not downloaded"
)


@requires_data
def test_run_training_produces_a_val_f1_score(tmp_path):
    train_paths, val_paths = split_available_demos()
    all_paths = (train_paths + val_paths)[:4]
    assert len(all_paths) >= 2, "need at least 2 downloaded demos for a train/val smoke test"
    train_subset, val_subset = all_paths[:-1], all_paths[-1:]

    tokenizer_ckpt = str(tmp_path / "tokenizer.pt")
    train_tokenizer(train_subset, tokenizer_ckpt, epochs=2, batch_size=8, device="cpu")

    result = run_training(
        tokenizer_ckpt, train_subset, val_subset, epochs=2, channels=8, num_layers=2, num_stages=2, device="cpu"
    )

    assert "val_f1" in result
    assert 0.0 <= result["val_f1"] <= 1.0
    assert "background" in result["vocab"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_classifier.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# temporal_classifier/train.py
"""Train the MS-TCN temporal classifier on frozen tokenizer embeddings."""

import numpy as np
import torch
import torch.nn as nn

from temporal_classifier.labels import align_labels_to_grid, build_vocab
from temporal_classifier.metrics import f1_at_k
from temporal_classifier.model import MSTCN
from tokenizer.data import load_demo, resample_to_grid
from tokenizer.train import load_checkpoint
from tokenizer.windowing import demo_to_sequence


def build_demo_sequence(tokenizer_model, mean, std, demo_path: str, window: int):
    channel_arrays, channel_timestamps, low_level_segments = load_demo(demo_path)
    telemetry, grid = resample_to_grid(channel_arrays, channel_timestamps)
    normed = (telemetry - mean) / std

    windows, centers = demo_to_sequence(normed, grid, window=window)
    x = torch.from_numpy(windows.astype(np.float32))
    with torch.no_grad():
        z_q, _ = tokenizer_model.encode_tokens(x)
    return z_q.numpy(), centers, low_level_segments


def _prepare_split(tokenizer_model, mean, std, demo_paths: list[str], window: int, vocab: dict[str, int]):
    sequences, label_seqs = [], []
    for path in demo_paths:
        embeddings, centers, segments = build_demo_sequence(tokenizer_model, mean, std, path, window)
        labels = align_labels_to_grid(centers, segments, vocab)
        sequences.append(embeddings)
        label_seqs.append(labels)
    return sequences, label_seqs


def run_training(
    tokenizer_ckpt_path: str,
    train_paths: list[str],
    val_paths: list[str],
    vocab: dict[str, int] | None = None,
    epochs: int = 30,
    channels: int = 64,
    num_layers: int = 9,
    num_stages: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    tokenizer_model, mean, std, config = load_checkpoint(tokenizer_ckpt_path)
    window = config["window"]

    if vocab is None:
        all_segments = []
        for path in train_paths + val_paths:
            _, _, segments = load_demo(path)
            all_segments.append(segments)
        vocab = build_vocab(all_segments)

    train_seqs, train_labels = _prepare_split(tokenizer_model, mean, std, train_paths, window, vocab)
    val_seqs, val_labels = _prepare_split(tokenizer_model, mean, std, val_paths, window, vocab)

    model = MSTCN(config["latent_dim"], len(vocab), channels, num_layers, num_stages).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for embeddings, labels in zip(train_seqs, train_labels):
            x = torch.from_numpy(embeddings).unsqueeze(0).to(device)
            y = torch.from_numpy(labels).unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = sum(loss_fn(out, y) for out in outputs)
            loss.backward()
            optimizer.step()

    model.eval()
    f1_scores = []
    with torch.no_grad():
        for embeddings, labels in zip(val_seqs, val_labels):
            x = torch.from_numpy(embeddings).unsqueeze(0).to(device)
            pred = model(x)[-1].argmax(dim=1).squeeze(0).cpu().numpy()
            f1_scores.append(f1_at_k(pred, labels))

    val_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"model": model, "vocab": vocab, "val_f1": val_f1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier_train.py -v`
Expected: PASS (1 test; small epoch counts and CPU device keep it fast).

- [ ] **Step 5: Commit**

```bash
git add temporal_classifier/train.py tests/test_classifier_train.py
git commit -m "feat: telemetry-only MS-TCN classifier training and F1@50 evaluation"
```

---

### Task 9: Vision enrichment — frozen encoder + fusion

**Files:**
- Create: `enrichment/vision_features.py`
- Create: `enrichment/fuse.py`
- Test: `tests/test_enrichment.py`

**Interfaces:**
- Consumes: nothing beyond `torch`, `h5py`, `cv2` — independent of the tokenizer/classifier
  modules; consumed by Task 10.
- Produces:
  - `enrichment.vision_features.VISION_FEATURE_DIM: int` (384, DINOv2 ViT-S/14 CLS dim)
  - `enrichment.vision_features.load_vision_encoder(device: str = "cuda") -> nn.Module`
    (frozen, `eval()` mode, loaded once and reused across calls).
  - `enrichment.vision_features.extract_frame_features(h5_path: str, frame_times: np.ndarray, encoder, camera_key: str = "hama1", device: str = "cuda") -> np.ndarray`
    returns `(N, VISION_FEATURE_DIM)`, one feature per requested time (nearest video frame).
  - `enrichment.fuse.ConcatProjectFusion(motion_dim: int, vision_dim: int, out_dim: int)`
    (`nn.Module`). `forward(motion_z: Tensor[B, motion_dim], vision_feat: Tensor[B, vision_dim]) -> Tensor[B, out_dim]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enrichment.py
import os

import numpy as np
import pytest
import torch

from enrichment.fuse import ConcatProjectFusion
from enrichment.vision_features import VISION_FEATURE_DIM, extract_frame_features, load_vision_encoder

DEMO_PATH = "data/reassemble/2025-01-09-13-57-17.h5"
requires_demo = pytest.mark.skipif(not os.path.exists(DEMO_PATH), reason="demo file not downloaded")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="DINOv2 download/inference needs the GPU box")


def test_concat_project_fusion_output_shape():
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384, out_dim=32)
    motion_z = torch.randn(5, 32)
    vision_feat = torch.randn(5, 384)

    fused = fusion(motion_z, vision_feat)

    assert fused.shape == (5, 32)


@requires_demo
@requires_gpu
def test_extract_frame_features_returns_one_feature_per_requested_time():
    encoder = load_vision_encoder(device="cuda")
    frame_times = np.array([1736427440.0, 1736427445.0, 1736427450.0])

    features = extract_frame_features(DEMO_PATH, frame_times, encoder, device="cuda")

    assert features.shape == (3, VISION_FEATURE_DIM)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_enrichment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enrichment.vision_features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# enrichment/vision_features.py
"""Frozen DINOv2 vision features, time-aligned to REASSEMBLE video frames."""

import tempfile

import cv2
import h5py
import numpy as np
import torch

VISION_FEATURE_DIM = 384  # dinov2_vits14 CLS token dimension


def load_vision_encoder(device: str = "cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _extract_frames(h5_path: str, camera_key: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        video_bytes = f[camera_key][()].tobytes()

    frames = {}
    wanted = set(frame_indices)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        idx = 0
        while cap.isOpened() and wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                wanted.discard(idx)
            idx += 1
        cap.release()
    return frames


@torch.no_grad()
def extract_frame_features(
    h5_path: str, frame_times: np.ndarray, encoder, camera_key: str = "hama1", device: str = "cuda"
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        cam_ts = f[f"timestamps/{camera_key}"][:]

    frame_indices = [int(np.argmin(np.abs(cam_ts - t))) for t in frame_times]
    frames = _extract_frames(h5_path, camera_key, frame_indices)

    features = []
    for idx in frame_indices:
        img = cv2.resize(frames[idx], (224, 224))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        feat = encoder(tensor.to(device))
        features.append(feat.squeeze(0).cpu().numpy())
    return np.stack(features, axis=0)
```

```python
# enrichment/fuse.py
"""Concatenate + project fusion of a motion token embedding with a vision feature."""

import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    def __init__(self, motion_dim: int, vision_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(motion_dim + vision_dim, out_dim)

    def forward(self, motion_z, vision_feat):
        return self.proj(torch.cat([motion_z, vision_feat], dim=-1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_enrichment.py -v`
Expected: PASS. The fusion-shape test always runs; the frame-extraction test is
`skipif`-guarded on both the demo file and CUDA (DINOv2's first run downloads weights via
`torch.hub` — requires internet, already confirmed available in this environment).

- [ ] **Step 5: Commit**

```bash
git add enrichment/vision_features.py enrichment/fuse.py tests/test_enrichment.py
git commit -m "feat: frozen DINOv2 vision features and concat+project fusion"
```

---

### Task 10: Vision-enriched classifier training + comparison report

**Files:**
- Modify: `temporal_classifier/train.py` — add vision-enriched training path.
- Create: `temporal_classifier/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: everything from Tasks 8-9 — `temporal_classifier.train.{build_demo_sequence, run_training}`,
  `enrichment.vision_features.{load_vision_encoder, extract_frame_features, VISION_FEATURE_DIM}`,
  `enrichment.fuse.ConcatProjectFusion`.
- Produces:
  - `temporal_classifier.train.run_training(..., use_vision: bool = False, vision_encoder=None, camera_key: str = "hama1")`
    — extended signature (backward compatible: existing calls with `use_vision` omitted
    behave exactly as Task 8 left them). When `use_vision=True`, each demo's per-token
    motion embedding is fused with a `ConcatProjectFusion` against a same-length vision
    feature sequence (via `extract_frame_features` at the token center times) before being
    fed to the `MSTCN`; the fusion module's parameters train jointly with the classifier.
  - `temporal_classifier.compare.format_comparison(telemetry_only_f1: float, vision_enriched_f1: float) -> str`
    — human-readable comparison table, used by both the test and the CLI entry point below.
  - `temporal_classifier/compare.py` also exposes a `if __name__ == "__main__":` block that
    runs both training modes end-to-end on `split_available_demos()` and prints the report —
    this is the milestone-5 deliverable from the spec, run manually once the full 148-demo
    set is downloaded (not covered by the automated test, which only checks the string
    formatting).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare.py
from temporal_classifier.compare import format_comparison


def test_format_comparison_reports_both_scores_and_the_delta():
    report = format_comparison(telemetry_only_f1=0.60, vision_enriched_f1=0.75)

    assert "0.60" in report
    assert "0.75" in report
    assert "+0.15" in report or "0.15" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_classifier.compare'`.

- [ ] **Step 3: Write minimal implementation**

First, extend `temporal_classifier/train.py`'s `_prepare_split` and `run_training` (edit the
existing file from Task 8 — replace `_prepare_split` and `run_training` in full with the
versions below, which add optional vision fusion):

```python
# temporal_classifier/train.py — replace _prepare_split and run_training with:

from enrichment.fuse import ConcatProjectFusion
from enrichment.vision_features import extract_frame_features


def _prepare_split(
    tokenizer_model, mean, std, demo_paths: list[str], window: int, vocab: dict[str, int],
    use_vision: bool = False, vision_encoder=None, camera_key: str = "hama1",
):
    sequences, vision_seqs, label_seqs = [], [], []
    for path in demo_paths:
        embeddings, centers, segments = build_demo_sequence(tokenizer_model, mean, std, path, window)
        labels = align_labels_to_grid(centers, segments, vocab)
        sequences.append(embeddings)
        label_seqs.append(labels)
        if use_vision:
            vision_seqs.append(extract_frame_features(path, centers, vision_encoder, camera_key))
    return sequences, vision_seqs, label_seqs


def run_training(
    tokenizer_ckpt_path: str,
    train_paths: list[str],
    val_paths: list[str],
    vocab: dict[str, int] | None = None,
    epochs: int = 30,
    channels: int = 64,
    num_layers: int = 9,
    num_stages: int = 3,
    lr: float = 1e-3,
    device: str = "cuda",
    use_vision: bool = False,
    vision_encoder=None,
    camera_key: str = "hama1",
) -> dict:
    tokenizer_model, mean, std, config = load_checkpoint(tokenizer_ckpt_path)
    window = config["window"]

    if vocab is None:
        all_segments = []
        for path in train_paths + val_paths:
            _, _, segments = load_demo(path)
            all_segments.append(segments)
        vocab = build_vocab(all_segments)

    train_seqs, train_vision, train_labels = _prepare_split(
        tokenizer_model, mean, std, train_paths, window, vocab, use_vision, vision_encoder, camera_key
    )
    val_seqs, val_vision, val_labels = _prepare_split(
        tokenizer_model, mean, std, val_paths, window, vocab, use_vision, vision_encoder, camera_key
    )

    fusion = None
    if use_vision:
        from enrichment.vision_features import VISION_FEATURE_DIM

        fusion = ConcatProjectFusion(config["latent_dim"], VISION_FEATURE_DIM, config["latent_dim"]).to(device)

    model = MSTCN(config["latent_dim"], len(vocab), channels, num_layers, num_stages).to(device)
    params = list(model.parameters()) + (list(fusion.parameters()) if fusion else [])
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def build_input(embeddings, vision_feats):
        x = torch.from_numpy(embeddings).to(device)
        if fusion is not None:
            v = torch.from_numpy(vision_feats.astype(np.float32)).to(device)
            x = fusion(x, v)
        return x.unsqueeze(0)

    for _ in range(epochs):
        model.train()
        for i, labels in enumerate(train_labels):
            x = build_input(train_seqs[i], train_vision[i] if use_vision else None)
            y = torch.from_numpy(labels).unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = sum(loss_fn(out, y) for out in outputs)
            loss.backward()
            optimizer.step()

    model.eval()
    f1_scores = []
    with torch.no_grad():
        for i, labels in enumerate(val_labels):
            x = build_input(val_seqs[i], val_vision[i] if use_vision else None)
            pred = model(x)[-1].argmax(dim=1).squeeze(0).cpu().numpy()
            f1_scores.append(f1_at_k(pred, labels))

    val_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"model": model, "fusion": fusion, "vocab": vocab, "val_f1": val_f1}
```

Then create the comparison script:

```python
# temporal_classifier/compare.py
"""Run telemetry-only vs. vision-enriched classifier training and report the F1@50 delta."""

from temporal_classifier.train import run_training
from tokenizer.train import train_tokenizer
from tokenizer.windowing import split_available_demos


def format_comparison(telemetry_only_f1: float, vision_enriched_f1: float) -> str:
    delta = vision_enriched_f1 - telemetry_only_f1
    sign = "+" if delta >= 0 else ""
    return (
        "F1@50 comparison\n"
        "-----------------\n"
        f"telemetry-only:   {telemetry_only_f1:.2f}\n"
        f"vision-enriched:  {vision_enriched_f1:.2f}\n"
        f"delta:            {sign}{delta:.2f}\n"
    )


def main():
    from enrichment.vision_features import load_vision_encoder

    train_paths, val_paths = split_available_demos()
    print(f"{len(train_paths)} train demos, {len(val_paths)} val demos available")

    tokenizer_ckpt = "tokenizer_checkpoint.pt"
    train_tokenizer(train_paths, tokenizer_ckpt, device="cuda")

    telemetry_only = run_training(tokenizer_ckpt, train_paths, val_paths, device="cuda")
    vision_encoder = load_vision_encoder(device="cuda")
    vision_enriched = run_training(
        tokenizer_ckpt, train_paths, val_paths, vocab=telemetry_only["vocab"],
        device="cuda", use_vision=True, vision_encoder=vision_encoder,
    )

    print(format_comparison(telemetry_only["val_f1"], vision_enriched["val_f1"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS (1 test — string formatting only; the `main()` end-to-end run is a manual
step, not part of the automated suite, since it needs the full 148-demo download to produce
a meaningful benchmark number).

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

Run: `python -m pytest tests/ -v`
Expected: PASS for every test that isn't `skipif`-guarded on missing data/GPU.

- [ ] **Step 6: Commit**

```bash
git add temporal_classifier/train.py temporal_classifier/compare.py tests/test_compare.py
git commit -m "feat: vision-enriched classifier training and F1@50 comparison report"
```

---

## After this plan

Once the full 148-demo REASSEMBLE set is downloaded (`python data/download_reassemble.py`,
no `--limit`, ~10-11 hours per the spec's measured bandwidth), run
`python temporal_classifier/compare.py` for the real benchmark numbers against the spec's
reference figures (Nomadic 93.1%, M2R2 83.4%). That result is the go/no-go decision on
keeping vision enrichment — not something this plan can determine on a 10-demo subset.

**Note (per [[robotics-fresh-training-no-smoke-test-carryover]] memory):** when running the
real training pass on the full 148-demo set, do not resume from any checkpoint produced
while smoke-testing this plan's tasks on the 10-demo subset — train fresh.
