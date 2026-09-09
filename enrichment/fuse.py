"""Concatenate + project fusion of a motion token embedding with a vision feature."""

import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    def __init__(self, motion_dim: int, vision_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(motion_dim + vision_dim, out_dim)

    def forward(self, motion_z, vision_feat):
        return self.proj(torch.cat([motion_z, vision_feat], dim=-1))
