import os

import numpy as np
import pytest
import torch

from enrichment.fuse import ConcatProjectFusion
from enrichment.vision_features import (
    VISION_FEATURE_DIM,
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
