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
