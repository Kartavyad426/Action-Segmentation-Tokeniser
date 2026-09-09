import os

import numpy as np
import pytest

from tokenizer.data import TELEMETRY_CHANNELS, load_demo, resample_to_grid
from tokenizer.windowing import (
    WindowedTelemetryDataset,
    compute_norm_stats,
    demo_to_sequence,
    split_available_demos,
    split_available_demos_3way,
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


def _write_splits(tmp_path, train_stems, test_stems):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train_split1.txt").write_text("\n".join(train_stems) + "\n")
    (splits / "test_split1.txt").write_text("\n".join(test_stems) + "\n")
    return splits


def _write_demos(tmp_path, stems):
    data = tmp_path / "demos"
    data.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (data / f"{s}.h5").touch()
    return data


def test_three_way_split_is_disjoint_and_covers_the_official_train_list(tmp_path):
    train_stems = [f"demo{i:03d}" for i in range(20)]
    test_stems = [f"held{i:03d}" for i in range(5)]
    splits = _write_splits(tmp_path, train_stems, test_stems)
    data = _write_demos(tmp_path, train_stems + test_stems)

    tr, va, te = split_available_demos_3way(data_dir=str(data), splits_dir=str(splits), val_fraction=0.25)

    assert set(tr) & set(va) == set()
    assert set(tr) | set(va) == {str(data / f"{s}.h5") for s in train_stems}
    assert len(te) == 5
    assert len(va) == 5


def test_three_way_split_never_puts_a_test_demo_in_train_or_val(tmp_path):
    train_stems = [f"demo{i:03d}" for i in range(20)]
    test_stems = [f"held{i:03d}" for i in range(5)]
    splits = _write_splits(tmp_path, train_stems, test_stems)
    data = _write_demos(tmp_path, train_stems + test_stems)

    tr, va, te = split_available_demos_3way(data_dir=str(data), splits_dir=str(splits))

    held = {str(data / f"{s}.h5") for s in test_stems}
    assert not (set(tr) & held)
    assert not (set(va) & held)
    assert set(te) == held


def test_val_assignment_does_not_shift_as_more_demos_finish_downloading(tmp_path):
    # The assignment is derived from the canonical split file, not from what happens to be
    # on disk -- otherwise a demo's train/val membership would change mid-download and the
    # validation set would silently stop being held out.
    train_stems = [f"demo{i:03d}" for i in range(20)]
    splits = _write_splits(tmp_path, train_stems, ["held000"])

    partial = _write_demos(tmp_path / "a", train_stems[:10])
    complete = _write_demos(tmp_path / "b", train_stems)

    _, va_partial, _ = split_available_demos_3way(data_dir=str(partial), splits_dir=str(splits), val_fraction=0.25)
    _, va_complete, _ = split_available_demos_3way(data_dir=str(complete), splits_dir=str(splits), val_fraction=0.25)

    partial_stems = {os.path.basename(p) for p in va_partial}
    complete_stems = {os.path.basename(p) for p in va_complete}
    assert partial_stems <= complete_stems, "val membership changed as demos arrived"


def test_three_way_split_is_deterministic_and_seed_controlled(tmp_path):
    train_stems = [f"demo{i:03d}" for i in range(20)]
    splits = _write_splits(tmp_path, train_stems, ["held000"])
    data = _write_demos(tmp_path, train_stems)

    a = split_available_demos_3way(data_dir=str(data), splits_dir=str(splits), seed=0)[1]
    b = split_available_demos_3way(data_dir=str(data), splits_dir=str(splits), seed=0)[1]
    c = split_available_demos_3way(data_dir=str(data), splits_dir=str(splits), seed=7)[1]

    assert a == b
    assert set(a) != set(c)
