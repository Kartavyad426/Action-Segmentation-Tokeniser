"""Turn resampled telemetry into fixed-length windows for tokenizer training/inference."""

import random
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


def split_available_demos_3way(
    data_dir: str = "data/reassemble",
    splits_dir: str = "data/splits_inspect",
    val_fraction: float = 0.2,
    seed: int = 0,
):
    """REASSEMBLE's official split1, with a validation set carved out of the train half.

    `test_split1` is the published held-out set and is what any number quoted against M2R2
    or Nomadic must be measured on -- so it is never touched during tuning. Hyperparameter
    selection uses the validation demos instead.

    The train/val assignment is computed from the canonical split file, not from what is
    currently on disk: otherwise a demo's membership would change as the dataset finishes
    downloading, and a demo held out in one run could be trained on in the next.
    """
    def read_stems(name: str) -> list[str]:
        with open(Path(splits_dir) / name) as f:
            return [line.strip() for line in f if line.strip()]

    train_stems = sorted(read_stems("train_split1.txt"))
    n_val = round(len(train_stems) * val_fraction)
    val_stems = set(random.Random(seed).sample(train_stems, n_val))

    def resolve(stems) -> list[str]:
        return [
            str(Path(data_dir) / f"{s}.h5")
            for s in sorted(stems)
            if (Path(data_dir) / f"{s}.h5").exists()
        ]

    return (
        resolve([s for s in train_stems if s not in val_stems]),
        resolve(val_stems),
        resolve(read_stems("test_split1.txt")),
    )
