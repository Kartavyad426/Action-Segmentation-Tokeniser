"""MS-TCN: multi-stage temporal convolutional network for per-token action classification."""

import torch
import torch.nn as nn


class _DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = torch.relu(self.conv(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class _SingleStageTCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: int, num_layers: int):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.layers = nn.ModuleList(_DilatedResidualLayer(2**i, channels) for i in range(num_layers))
        self.out_conv = nn.Conv1d(channels, num_classes, kernel_size=1)

    def forward(self, x):  # x: (B, C, T)
        h = self.in_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_conv(h)  # (B, num_classes, T)


class MSTCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: int = 64, num_layers: int = 9, num_stages: int = 3):
        super().__init__()
        self.stage1 = _SingleStageTCN(in_channels, num_classes, channels, num_layers)
        self.stages = nn.ModuleList(
            _SingleStageTCN(num_classes, num_classes, channels, num_layers) for _ in range(num_stages - 1)
        )

    def forward(self, x):  # x: (B, T, in_channels)
        x = x.transpose(1, 2)  # (B, in_channels, T)
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(torch.softmax(out, dim=1))
            outputs.append(out)
        return outputs
