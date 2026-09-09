"""MS-TCN's training objective: cross-entropy plus a truncated MSE smoothing term."""

import torch
import torch.nn.functional as F


def truncated_mse_smoothing_loss(logits: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """Penalizes frame-to-frame change in the predicted log-probabilities (Farha & Gall,
    MS-TCN 2019).

    Cross-entropy scores every token independently, so nothing in it objects to a prediction
    that flips class back and forth between neighbouring tokens. Segmental F1@50 objects
    strongly: a fragmented prediction produces many short segments that fail the 0.5 IoU
    match and count as false positives. This term supplies the missing pressure.

    Truncation at `tau` is what keeps it from fighting the cross-entropy: a real action
    boundary *should* produce a large jump in log-probabilities, so an untruncated penalty
    would punish exactly the transitions we want the model to predict. Clamping the squared
    difference at `tau**2` caps what any single frame pair can contribute, so genuine
    boundaries cost a bounded amount while sustained flicker -- many small penalties in a
    row -- still accumulates.

    The previous frame is detached: the loss pulls the current frame toward the previous
    one, rather than dragging the previous frame backwards to meet it.
    """
    log_p = F.log_softmax(logits, dim=1)
    delta = (log_p[:, :, 1:] - log_p.detach()[:, :, :-1]) ** 2
    return torch.clamp(delta, min=0.0, max=tau**2).mean()


def mstcn_loss(
    outputs: list[torch.Tensor],
    targets: torch.Tensor,
    smoothing_weight: float = 0.15,
    tau: float = 4.0,
) -> torch.Tensor:
    """Summed over every stage's output, as MS-TCN trains each stage against the labels."""
    total = 0.0
    for out in outputs:
        total = total + F.cross_entropy(out, targets)
        if smoothing_weight:
            total = total + smoothing_weight * truncated_mse_smoothing_loss(out, tau)
    return total
