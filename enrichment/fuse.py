"""Concatenate + project fusion of a motion token embedding with a vision feature."""

import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    """Fuses a motion token embedding with a time-aligned vision feature.

    Each branch is LayerNorm'd *separately, before* the concatenation. This is load-bearing:
    measured on real data, the motion embedding enters at per-vector norm ~1.83 while the
    DINOv2 CLS token enters at ~47.1. Since `nn.Linear` draws all of its input weights from
    one distribution -- it has no idea which columns belong to which modality -- each
    branch's contribution to the output scales with that branch's vector norm, so the 25.7x
    input gap is a 25.7x branch-dominance gap at initialization and in the gradients.
    LayerNorm pins each branch's norm to exactly sqrt(dim) regardless of what came in,
    leaving only the sqrt(384/32) ~= 3.5x that is pure dimensionality, which the learnable
    affine gains can subsequently re-weight.

    Normalizing the *concatenated* 416-dim vector instead would not work: one shared scalar
    applied to both halves leaves their ratio exactly unchanged.
    """

    def __init__(
        self,
        motion_dim: int,
        vision_dim: int,
        out_dim: int = 128,
        normalize_motion: bool = True,
        normalize_vision: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.motion_norm = nn.LayerNorm(motion_dim) if normalize_motion else nn.Identity()
        self.vision_norm = nn.LayerNorm(vision_dim) if normalize_vision else nn.Identity()
        self.proj = nn.Linear(motion_dim + vision_dim, out_dim)

    def normalized_branches(self, motion_z, vision_feat):
        """The two branches exactly as they enter the concatenation. Exposed so tests can
        assert on their relative scale, which is the property that was broken."""
        return self.motion_norm(motion_z), self.vision_norm(vision_feat)

    def forward(self, motion_z, vision_feat):
        motion_z, vision_feat = self.normalized_branches(motion_z, vision_feat)
        return self.proj(torch.cat([motion_z, vision_feat], dim=-1))
