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
