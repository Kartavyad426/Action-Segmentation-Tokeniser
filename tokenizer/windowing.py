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
