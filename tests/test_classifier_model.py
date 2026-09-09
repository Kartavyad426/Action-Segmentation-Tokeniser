import torch

from temporal_classifier.model import MSTCN


def test_forward_returns_one_output_per_stage_with_correct_shape():
    model = MSTCN(in_channels=32, num_classes=6, channels=16, num_layers=4, num_stages=3)
    x = torch.randn(2, 20, 32)

    outputs = model(x)

    assert len(outputs) == 3
    for out in outputs:
        assert out.shape == (2, 6, 20)


def test_gradient_flows_to_first_stage_from_last_stage_loss():
    model = MSTCN(in_channels=8, num_classes=3, channels=8, num_layers=2, num_stages=2)
    x = torch.randn(1, 10, 8)

    outputs = model(x)
    loss = outputs[-1].sum()
    loss.backward()

    grad = model.stage1.in_conv.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0
