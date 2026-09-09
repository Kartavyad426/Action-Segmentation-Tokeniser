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
