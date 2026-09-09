import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from enrichment.fuse import ConcatProjectFusion
from enrichment.vision_features import (
    VISION_FEATURE_DIM,
    _iter_wanted_frames,
    _preprocess_for_dinov2,
    extract_frame_features,
    load_vision_encoder,
)

DEMO_PATH = "data/reassemble/2025-01-09-13-57-17.h5"
requires_demo = pytest.mark.skipif(not os.path.exists(DEMO_PATH), reason="demo file not downloaded")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="DINOv2 download/inference needs the GPU box")


def test_concat_project_fusion_output_shape():
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384, out_dim=32)
    motion_z = torch.randn(5, 32)
    vision_feat = torch.randn(5, 384)

    fused = fusion(motion_z, vision_feat)

    assert fused.shape == (5, 32)


def test_preprocess_for_dinov2_resizes_shortest_side_then_center_crops():
    # A non-square image (taller than wide) should have its shortest side (width)
    # resized to 256, then a 224x224 center crop taken, preserving aspect ratio.
    img = np.zeros((480, 320, 3), dtype=np.uint8)

    out = _preprocess_for_dinov2(img, resize_short=256, crop=224)

    assert out.shape == (224, 224, 3)


def test_preprocess_for_dinov2_normalizes_with_imagenet_stats():
    # An all-zero image, after /255 normalization, should map to -mean/std per channel.
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    out = _preprocess_for_dinov2(img)

    expected = -np.array([0.485, 0.456, 0.406], dtype=np.float32) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    assert np.allclose(out[0, 0], expected, atol=1e-5)
    # confirms it's not raw [0,1]-scaled data with no normalization
    assert not np.allclose(out, 0.0)


def test_preprocess_for_dinov2_does_not_squash_aspect_ratio():
    # A wide image with a distinct vertical stripe pattern: naive resize-to-224x224
    # (ignoring aspect ratio) would distort the stripe width differently than a
    # resize-shortest-side + center-crop pipeline. Sanity check the crop lands
    # in-bounds and preserves recognizable structure (non-uniform output).
    img = np.zeros((224, 640, 3), dtype=np.uint8)
    img[:, 300:340, :] = 255  # a stripe near the horizontal center

    out = _preprocess_for_dinov2(img, resize_short=256, crop=224)

    assert out.shape == (224, 224, 3)
    # the stripe should still be visible (not averaged away) somewhere in the crop
    assert out.max() > out.min()


@requires_demo
@requires_gpu
def test_extract_frame_features_returns_one_feature_per_requested_time():
    encoder = load_vision_encoder(device="cuda")
    frame_times = np.array([1736427440.0, 1736427445.0, 1736427450.0])

    features = extract_frame_features(DEMO_PATH, frame_times, encoder, device="cuda")

    assert features.shape == (3, VISION_FEATURE_DIM)


def _realistic_branch_inputs(n_tokens: int = 64, seed: int = 0):
    """Motion/vision vectors matching the scales measured on real data with a trained
    tokenizer: motion per-vector norm ~1.83, DINOv2 CLS norm ~47.1 where most of that
    magnitude is a large offset shared across frames, with the per-frame informative
    signal a small residual riding on top of it."""
    g = torch.Generator().manual_seed(seed)

    motion = torch.randn(n_tokens, 32, generator=g)
    motion = motion / motion.norm(dim=-1, keepdim=True) * 1.83

    shared_offset = torch.randn(384, generator=g)
    shared_offset = shared_offset / shared_offset.norm() * 47.0
    per_frame = torch.randn(n_tokens, 384, generator=g)
    per_frame = per_frame / per_frame.norm(dim=-1, keepdim=True) * 4.0
    vision = shared_offset + per_frame

    return motion, vision


def test_realistic_inputs_reproduce_the_measured_scale_gap():
    # Guards the fixture itself: if this drifts, the tests below stop testing the real bug.
    motion, vision = _realistic_branch_inputs()

    assert 1.7 < motion.norm(dim=-1).mean() < 2.0
    assert 46.0 < vision.norm(dim=-1).mean() < 48.0


def test_fusion_normalizes_branches_to_comparable_vector_norms():
    # The bug: motion enters at norm ~1.83 against vision's ~47.1, a 25.7x gap. Because
    # nn.Linear draws all 416 input weights from one distribution, that ratio is exactly
    # the branch-dominance ratio at init -- the motion signal is swamped.
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384)
    motion, vision = _realistic_branch_inputs()

    motion_n, vision_n = fusion.normalized_branches(motion, vision)

    ratio = vision_n.norm(dim=-1).mean() / motion_n.norm(dim=-1).mean()
    assert ratio < 5.0, f"vision branch enters {ratio:.1f}x louder than motion"


def test_fusion_normalizes_branches_to_comparable_per_element_scale():
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384)
    motion, vision = _realistic_branch_inputs()

    motion_n, vision_n = fusion.normalized_branches(motion, vision)

    motion_rms = motion_n.pow(2).mean().sqrt()
    vision_rms = vision_n.pow(2).mean().sqrt()
    assert 0.67 < (vision_rms / motion_rms) < 1.5


def test_fusion_output_is_not_dominated_by_the_vision_branch():
    # The consequence that actually matters: each branch's contribution to the projected
    # output scales with its vector norm, so a 25.7x input gap is a 25.7x output gap.
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384)
    motion, vision = _realistic_branch_inputs()

    motion_n, vision_n = fusion.normalized_branches(motion, vision)
    w = fusion.proj.weight
    motion_contribution = motion_n @ w[:, :32].t()
    vision_contribution = vision_n @ w[:, 32:].t()

    ratio = vision_contribution.std() / motion_contribution.std()
    assert ratio < 5.0, f"vision contributes {ratio:.1f}x more than motion to the fused output"


def test_fusion_removes_a_constant_offset_shared_across_frames():
    # A vision feature that is mostly a frame-invariant offset carries its information in
    # the residual. After normalization the fused output must still track that residual
    # rather than being dominated by the constant.
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384)
    motion, vision = _realistic_branch_inputs()

    _, vision_n = fusion.normalized_branches(motion, vision)

    across_frame_variation = vision_n.std(dim=0).mean()
    within_frame_scale = vision_n.pow(2).mean().sqrt()
    assert across_frame_variation / within_frame_scale > 0.05


def test_fusion_out_dim_defaults_to_128_independently_of_motion_dim():
    # out_dim was pinned to the tokenizer's latent_dim, coupling two unrelated
    # hyperparameters and squeezing 416 dims through a 32-dim bottleneck.
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384)

    assert fusion.out_dim == 128
    assert fusion(torch.randn(5, 32), torch.randn(5, 384)).shape == (5, 128)


def test_fusion_out_dim_is_settable():
    fusion = ConcatProjectFusion(motion_dim=32, vision_dim=384, out_dim=64)

    assert fusion.out_dim == 64
    assert fusion(torch.randn(5, 32), torch.randn(5, 384)).shape == (5, 64)


class _CountingCapture:
    """Stand-in for cv2.VideoCapture that records how many frames were decoded."""

    def __init__(self, n_frames: int = 200):
        self.reads = 0
        self.n_frames = n_frames

    def isOpened(self):
        return True

    def read(self):
        if self.reads >= self.n_frames:
            return False, None
        self.reads += 1
        return True, np.zeros((64, 64, 3), dtype=np.uint8)

    def release(self):
        pass


class _CountingEncoder:
    """Stand-in for DINOv2 that records the batch size of every call."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, tensor):
        self.batch_sizes.append(tensor.shape[0])
        return torch.zeros(tensor.shape[0], VISION_FEATURE_DIM)

    @property
    def calls(self):
        return len(self.batch_sizes)


def test_iter_wanted_frames_yields_before_decoding_the_whole_video():
    # The original implementation decoded every wanted frame into a dict before returning
    # anything -- measured ~3.5GB for one demo's worth of tokens. Streaming means the first
    # frame comes back after decoding exactly one frame, so peak memory is bounded by the
    # batch, not by the demo length.
    cap = _CountingCapture(n_frames=200)

    with patch("enrichment.vision_features.cv2.VideoCapture", return_value=cap):
        gen = _iter_wanted_frames("ignored.mp4", {0, 199})
        idx, frame = next(gen)

    assert idx == 0
    assert cap.reads == 1, f"decoded {cap.reads} frames before yielding the first one"


def test_iter_wanted_frames_stops_once_every_wanted_frame_is_seen():
    cap = _CountingCapture(n_frames=200)

    with patch("enrichment.vision_features.cv2.VideoCapture", return_value=cap):
        got = list(_iter_wanted_frames("ignored.mp4", {0, 4}))

    assert [idx for idx, _ in got] == [0, 4]
    assert cap.reads == 5, "kept decoding past the last wanted frame"


@requires_demo
def test_extract_frame_features_encodes_in_batches(tmp_path):
    # One encoder call per frame wastes GPU time; batching is what makes the full 148-demo
    # run affordable.
    encoder = _CountingEncoder()
    frame_times = np.linspace(1736427440.0, 1736427460.0, 10)

    features = extract_frame_features(
        DEMO_PATH, frame_times, encoder, device="cpu", batch_size=4
    )

    assert features.shape == (10, VISION_FEATURE_DIM)
    assert encoder.batch_sizes == [4, 4, 2], f"expected batches of 4, got {encoder.batch_sizes}"


@requires_demo
def test_extract_frame_features_reuses_a_disk_cache_instead_of_re_encoding(tmp_path):
    # The spec calls vision extraction "frozen, one-time, cacheable" but every
    # run_training(use_vision=True) re-decoded and re-encoded every frame from scratch.
    frame_times = np.linspace(1736427440.0, 1736427460.0, 6)

    first_encoder = _CountingEncoder()
    first = extract_frame_features(
        DEMO_PATH, frame_times, first_encoder, device="cpu", cache_dir=str(tmp_path)
    )

    second_encoder = _CountingEncoder()
    second = extract_frame_features(
        DEMO_PATH, frame_times, second_encoder, device="cpu", cache_dir=str(tmp_path)
    )

    assert first_encoder.calls > 0
    assert second_encoder.calls == 0, "cache hit should not touch the encoder at all"
    assert np.array_equal(first, second)


@requires_demo
def test_vision_cache_distinguishes_different_frame_requests(tmp_path):
    # A cache keyed only on the demo would serve stale features after a window-config change.
    encoder = _CountingEncoder()
    extract_frame_features(
        DEMO_PATH, np.linspace(1736427440.0, 1736427450.0, 4), encoder,
        device="cpu", cache_dir=str(tmp_path),
    )
    calls_after_first = encoder.calls

    extract_frame_features(
        DEMO_PATH, np.linspace(1736427450.0, 1736427460.0, 4), encoder,
        device="cpu", cache_dir=str(tmp_path),
    )

    assert encoder.calls > calls_after_first, "different frame times must miss the cache"


@requires_demo
def test_extract_frame_features_handles_a_frame_count_that_divides_evenly_by_the_batch():
    # The trailing partial batch is the common case; an exact multiple leaves an empty
    # buffer at the end and must not attempt to encode it.
    encoder = _CountingEncoder()
    frame_times = np.linspace(1736427440.0, 1736427460.0, 8)

    features = extract_frame_features(DEMO_PATH, frame_times, encoder, device="cpu", batch_size=4)

    assert features.shape == (8, VISION_FEATURE_DIM)
    assert encoder.batch_sizes == [4, 4]


@requires_demo
def test_extract_frame_features_handles_repeated_frame_requests():
    # Two token centers can land on the same video frame; the frame is encoded once but
    # must still appear at both output positions.
    encoder = _CountingEncoder()
    t = 1736427440.0
    features = extract_frame_features(DEMO_PATH, np.array([t, t, t]), encoder, device="cpu")

    assert features.shape == (3, VISION_FEATURE_DIM)
    assert encoder.batch_sizes == [1], "the same frame should only be encoded once"


class _DeterministicEncoder:
    """Content-dependent and deterministic, so pooled features can be checked against the
    mean of the individually-extracted frames they should be pooling."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, tensor):
        self.batch_sizes.append(tensor.shape[0])
        return tensor.mean(dim=(2, 3)).repeat(1, VISION_FEATURE_DIM // 3)

    @property
    def n_frames(self):
        return sum(self.batch_sizes)


@requires_demo
def test_window_pooling_encodes_more_frames_than_tokens():
    # Default sampling takes one frame per token and discards ~78% of the video. With
    # window_s set, every frame inside the token's window is encoded and averaged.
    times = np.linspace(1736427440.0, 1736427445.0, 4)

    nearest = _DeterministicEncoder()
    extract_frame_features(DEMO_PATH, times, nearest, device="cpu")

    pooled = _DeterministicEncoder()
    extract_frame_features(DEMO_PATH, times, pooled, device="cpu", window_s=0.15)

    assert nearest.n_frames == 4
    assert pooled.n_frames > 4, "window pooling should encode more than one frame per token"


@requires_demo
def test_window_pooled_feature_equals_the_mean_of_its_windows_frames():
    import h5py

    center = 1736427442.0
    window_s = 0.15
    with h5py.File(DEMO_PATH, "r") as f:
        cam_ts = f["timestamps/hama1"][:]
    in_window = cam_ts[(cam_ts >= center - window_s / 2) & (cam_ts <= center + window_s / 2)]
    assert len(in_window) > 1, "fixture assumes several frames land inside one window"

    pooled = extract_frame_features(
        DEMO_PATH, np.array([center]), _DeterministicEncoder(), device="cpu", window_s=window_s
    )
    individually = extract_frame_features(
        DEMO_PATH, in_window, _DeterministicEncoder(), device="cpu"
    )

    assert np.allclose(pooled[0], individually.mean(axis=0), atol=1e-5)


@requires_demo
def test_window_pooling_uses_a_different_cache_entry_than_nearest_frame(tmp_path):
    # Same demo, same token centers, different aggregation -- must not collide.
    times = np.linspace(1736427440.0, 1736427445.0, 4)

    extract_frame_features(DEMO_PATH, times, _DeterministicEncoder(), device="cpu", cache_dir=str(tmp_path))
    second = _DeterministicEncoder()
    extract_frame_features(DEMO_PATH, times, second, device="cpu", cache_dir=str(tmp_path), window_s=0.15)

    assert second.n_frames > 0, "pooled request wrongly hit the nearest-frame cache entry"
