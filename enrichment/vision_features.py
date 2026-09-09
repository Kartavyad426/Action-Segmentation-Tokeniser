"""Frozen DINOv2 vision features, time-aligned to REASSEMBLE video frames.

Extraction is frozen and one-time, so it streams and caches: frames are decoded one at a
time, encoded in batches, and discarded, and the resulting features are optionally persisted
to disk so repeated `run_training(use_vision=True)` calls never re-encode the same frames.
"""

import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

VISION_FEATURE_DIM = 384  # dinov2_vits14 CLS token dimension

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_vision_encoder(device: str = "cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _iter_wanted_frames(video_path: str, wanted: set[int]):
    """Yields (index, RGB frame) for each wanted frame, decoding lazily and stopping as
    soon as the last wanted frame has been seen."""
    remaining = set(wanted)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    try:
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in remaining:
                remaining.discard(idx)
                yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


@contextmanager
def _video_file(h5_path: str, camera_key: str):
    """Materializes the demo's encoded video to a temp file. Only the compressed bytes are
    held, never the decoded frames."""
    with h5py.File(h5_path, "r") as f:
        video_bytes = f[camera_key][()].tobytes()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        del video_bytes
        yield tmp.name


def _cache_path(cache_dir: str, h5_path: str, camera_key: str, groups: list[list[int]]) -> Path:
    """Keyed on the demo, the camera, and the exact per-token frame grouping -- so a change
    to the window config (which moves every token center) or to the aggregation mode misses
    the cache rather than silently serving features aligned to the old grid."""
    payload = b"|".join(np.asarray(g, dtype=np.int64).tobytes() for g in groups)
    key = hashlib.sha256(payload).hexdigest()[:16]
    return Path(cache_dir) / f"{Path(h5_path).stem}.{camera_key}.{key}.npy"


def _frame_groups(cam_ts: np.ndarray, frame_times: np.ndarray, window_s: float | None) -> list[list[int]]:
    """The video frames backing each token. One nearest frame by default; every frame inside
    the token's window when `window_s` is set, which are then mean-pooled."""
    if window_s is None:
        return [[int(np.argmin(np.abs(cam_ts - t)))] for t in frame_times]

    groups = []
    for t in frame_times:
        inside = np.flatnonzero((cam_ts >= t - window_s / 2) & (cam_ts <= t + window_s / 2))
        if len(inside) == 0:  # window fell between frames (or outside the video)
            inside = [int(np.argmin(np.abs(cam_ts - t)))]
        groups.append([int(i) for i in inside])
    return groups


def _encode_batch(batch: list[np.ndarray], encoder, device: str) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(batch, axis=0)).permute(0, 3, 1, 2).float()
    return np.asarray(encoder(tensor.to(device)).cpu())


def _preprocess_for_dinov2(img: np.ndarray, resize_short: int = 256, crop: int = 224) -> np.ndarray:
    """Resize shortest side to `resize_short`, center-crop to `crop`x`crop`, and
    normalize with ImageNet mean/std, matching DINOv2's standard preprocessing."""
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = resize_short, int(round(w * resize_short / h))
    else:
        new_h, new_w = int(round(h * resize_short / w)), resize_short
    resized = cv2.resize(img, (new_w, new_h))

    top = (new_h - crop) // 2
    left = (new_w - crop) // 2
    cropped = resized[top : top + crop, left : left + crop]

    normed = (cropped.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return normed


@torch.no_grad()
def extract_frame_features(
    h5_path: str,
    frame_times: np.ndarray,
    encoder,
    camera_key: str = "hama1",
    device: str = "cuda",
    batch_size: int = 32,
    cache_dir: str | None = None,
    window_s: float | None = None,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        cam_ts = f[f"timestamps/{camera_key}"][:]

    groups = _frame_groups(cam_ts, frame_times, window_s)
    wanted = sorted({i for g in groups for i in g})

    cache_file = _cache_path(cache_dir, h5_path, camera_key, groups) if cache_dir else None
    if cache_file is not None and cache_file.exists():
        return np.load(cache_file)

    by_index: dict[int, np.ndarray] = {}
    batch: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush():
        if not batch:
            return
        for i, feat in zip(batch_indices, _encode_batch(batch, encoder, device)):
            by_index[i] = feat
        batch.clear()
        batch_indices.clear()

    with _video_file(h5_path, camera_key) as video_path:
        for idx, frame in _iter_wanted_frames(video_path, set(wanted)):
            batch.append(_preprocess_for_dinov2(frame))
            batch_indices.append(idx)
            if len(batch) == batch_size:
                flush()
        flush()

    features = np.stack(
        [np.mean([by_index[i] for i in g], axis=0) for g in groups], axis=0
    ).astype(np.float32)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, features)
    return features
