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
