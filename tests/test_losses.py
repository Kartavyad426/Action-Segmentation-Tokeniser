import torch

from temporal_classifier.losses import mstcn_loss, truncated_mse_smoothing_loss


def _steady_logits(t=50, c=5):
    # one confident class held for the whole sequence: no frame-to-frame change at all
    logits = torch.full((1, c, t), -5.0)
    logits[:, 2, :] = 5.0
    return logits


def _flickering_logits(t=50, c=5):
    # prediction flips class every frame -- the over-segmentation the loss exists to punish
    logits = torch.full((1, c, t), -5.0)
    for i in range(t):
        logits[:, i % 2, i] = 5.0
    return logits


def test_smoothing_loss_is_zero_for_a_prediction_that_never_changes():
    assert truncated_mse_smoothing_loss(_steady_logits()).item() == 0.0


def test_smoothing_loss_punishes_a_flickering_prediction():
    steady = truncated_mse_smoothing_loss(_steady_logits())
    flickering = truncated_mse_smoothing_loss(_flickering_logits())

    assert flickering > steady
    assert flickering > 1.0


def _one_switch(magnitude, t=50):
    # two classes so that a change in magnitude moves only the switching classes' log-probs
    logits = torch.full((1, 2, t), -magnitude)
    logits[:, 1, :t // 2] = magnitude
    logits[:, 0, t // 2:] = magnitude
    return logits


def test_smoothing_loss_is_truncated_so_one_huge_jump_cannot_dominate():
    # Without truncation a single enormous log-prob jump would swamp the batch, and a real
    # action boundary is exactly such a jump. tau=4 clamps the squared difference at 16, so
    # a 100x larger swing costs the same.
    big = _one_switch(5.0)
    huge = _one_switch(500.0)

    assert torch.isclose(
        truncated_mse_smoothing_loss(big), truncated_mse_smoothing_loss(huge), atol=1e-4
    )


def test_smoothing_loss_does_not_backpropagate_through_the_previous_frame():
    # MS-TCN detaches the t-1 term: the loss should pull the current frame toward the
    # previous one, not drag the previous frame backwards to meet it.
    logits = torch.randn(1, 4, 2, requires_grad=True)

    truncated_mse_smoothing_loss(logits).backward()

    assert torch.allclose(logits.grad[:, :, 0], torch.zeros(4)), "gradient leaked into frame t-1"
    assert not torch.allclose(logits.grad[:, :, 1], torch.zeros(4))


def test_mstcn_loss_adds_smoothing_to_cross_entropy_across_every_stage():
    torch.manual_seed(0)
    outputs = [torch.randn(1, 5, 30, requires_grad=True) for _ in range(3)]
    targets = torch.randint(0, 5, (1, 30))

    with_smoothing = mstcn_loss(outputs, targets, smoothing_weight=0.15)
    without = mstcn_loss(outputs, targets, smoothing_weight=0.0)

    assert with_smoothing > without
    expected = sum(0.15 * truncated_mse_smoothing_loss(o) for o in outputs)
    assert torch.isclose(with_smoothing - without, expected, atol=1e-5)


def test_mstcn_loss_with_zero_weight_is_exactly_cross_entropy():
    torch.manual_seed(0)
    outputs = [torch.randn(1, 5, 30) for _ in range(2)]
    targets = torch.randint(0, 5, (1, 30))

    ce = sum(torch.nn.functional.cross_entropy(o, targets) for o in outputs)

    assert torch.isclose(mstcn_loss(outputs, targets, smoothing_weight=0.0), ce, atol=1e-6)
